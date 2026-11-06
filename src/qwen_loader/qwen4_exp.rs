//! Lecture d'un checkpoint `qwen4_exp` converti en MLX affine (`convert-qwen4exp`).
//!
//! Lecture bornée par `pread` : seules les plages demandées sont lues, les
//! affines sont déquantifiés à la demande (matrice dense, bloc de lignes,
//! lignes choisies ou un expert d'une pile `switch_mlp`). L'ouverture vérifie
//! la config, l'inventaire exact des tenseurs, les triplets
//! weight/scales/biases et les bits, la table n-gram et ses paramètres de
//! hachage ; la tête MTP n'est validée que sur ses métadonnées.

use std::collections::{BTreeSet, HashMap};
use std::fs::File;
use std::ops::Range;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use safetensors::Dtype;

use super::{read_shard_header, ShardHeader, ShardTensorEntry};
use crate::qwen4_exp::ngram::{head_vocab_sizes, layer_multipliers, NGRAM_SEED};
use crate::safetensor::{bytes_to_dense_f32, f8_e4m3_to_f32};
use crate::{dequantize_affine_u32, AffineQuantizedTensor, InferError, Result, Tensor};

const TRUNK: &str = "language_model.model.";
/// Borne des dimensions de config (2^20) : les produits restent représentables.
const MAX_DIM: usize = 1 << 20;
const LM_HEAD: &str = "language_model.lm_head";

/// Type d'une couche du tronc.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qwen4ExpLayerKind {
    /// Gated DeltaNet.
    Linear,
    /// Attention QSA (indexeur + attention GQA à gate sigmoïde).
    Full,
}

/// Config texte `qwen4_exp`, réduite à ce que le forward de référence utilise.
#[derive(Clone, Debug)]
pub struct Qwen4ExpConfig {
    pub hidden: usize,
    pub layer_kinds: Vec<Qwen4ExpLayerKind>,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub rope_theta: f64,
    pub linear_key_heads: usize,
    pub linear_value_heads: usize,
    pub linear_key_dim: usize,
    pub linear_value_dim: usize,
    pub conv_kernel: usize,
    pub experts: usize,
    pub top_k: usize,
    pub moe_inter: usize,
    pub shared_inter: usize,
    pub norm_topk_prob: bool,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_budget: usize,
    pub indexer_compress: usize,
    pub indexer_head_dim: usize,
    pub indexer_heads: usize,
    pub ple_layer: usize,
    pub ple_dim: usize,
    pub ple_conv_kernel: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ngram_vocab_base: u64,
    pub vocab: usize,
    pub eos: u64,
    pub rms_eps: f32,
    /// Gate de sortie DeltaNet : `σ(z)` (Flash-Next) ou `SiLU(z)`.
    pub sigmoid_output_gate: bool,
    pub mtp_layers: usize,
}

/// Chemin de la table n-gram désignée par `ngram/meta.json` : relatif et sous
/// `ngram/`. Un chemin absolu ou contenant `..` échapperait à la provenance.
pub(crate) fn ngram_table_path(dir: &Path, file: &str) -> Result<PathBuf> {
    let relative = Path::new(file);
    let contained = relative.components().next().is_some()
        && relative
            .components()
            .all(|c| matches!(c, std::path::Component::Normal(_)));
    if !contained {
        return Err(cfg_err(format!(
            "ngram/meta.json: file {file:?} hors de ngram/"
        )));
    }
    Ok(dir.join("ngram").join(relative))
}

fn cfg_err(message: impl Into<String>) -> InferError {
    InferError::Config(format!("qwen4_exp: {}", message.into()))
}

fn field<'a>(text: &'a serde_json::Value, key: &str) -> Result<&'a serde_json::Value> {
    text.get(key)
        .ok_or_else(|| cfg_err(format!("{key} absent")))
}

fn usize_of(text: &serde_json::Value, key: &str) -> Result<usize> {
    field(text, key)?
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
        .ok_or_else(|| cfg_err(format!("{key} entier attendu")))
}

fn usize_or(text: &serde_json::Value, key: &str, default: usize) -> Result<usize> {
    match text.get(key) {
        None | Some(serde_json::Value::Null) => Ok(default),
        Some(_) => usize_of(text, key),
    }
}

impl Qwen4ExpConfig {
    /// Lit la config (racine ou `text_config`) et refuse seulement les
    /// invariants supposés par le code (cf. `validate_architecture` amont).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un champ manque ou viole un invariant.
    pub fn from_json(config: &serde_json::Value) -> Result<Self> {
        let model_type = config.get("model_type").and_then(|v| v.as_str());
        if model_type != Some("qwen4_exp") && model_type != Some("qwen4_exp_text") {
            return Err(cfg_err(format!("model_type={model_type:?}")));
        }
        let text = config.get("text_config").unwrap_or(config);
        let layers = usize_of(text, "num_hidden_layers")?;
        let kinds = field(text, "layer_types")?
            .as_array()
            .ok_or_else(|| cfg_err("layer_types non liste"))?
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(Qwen4ExpLayerKind::Linear),
                Some("full_attention" | "indexed_attention") => Ok(Qwen4ExpLayerKind::Full),
                other => Err(cfg_err(format!("type de couche {other:?}"))),
            })
            .collect::<Result<Vec<_>>>()?;
        if kinds.len() != layers || layers == 0 {
            return Err(cfg_err(format!(
                "{} types pour {layers} couches",
                kinds.len()
            )));
        }
        let head_dim = usize_of(text, "head_dim")?;
        let rope = text.get("rope_parameters");
        let partial = rope
            .and_then(|r| r.get("partial_rotary_factor"))
            .or_else(|| text.get("partial_rotary_factor"))
            .and_then(|v| v.as_f64())
            .unwrap_or(1.0);
        let rope_theta = rope
            .and_then(|r| r.get("rope_theta"))
            .or_else(|| text.get("rope_theta"))
            .and_then(|v| v.as_f64())
            .ok_or_else(|| cfg_err("rope_theta absent"))?;
        // `int(head_dim * partial)` amont ; arrondi au pair inférieur pour la RoPE.
        let rotary_dim = ((head_dim as f64 * partial) as usize) & !1;
        let ple_ids = field(text, "ple_layer_ids")?
            .as_array()
            .ok_or_else(|| cfg_err("ple_layer_ids non liste"))?;
        let ple_layer = match ple_ids[..] {
            [ref id] => id
                .as_u64()
                .filter(|id| *id >= 1 && *id as usize <= layers)
                .map(|id| id as usize - 1)
                .ok_or_else(|| cfg_err(format!("ple_layer_ids={ple_ids:?}")))?,
            _ => {
                return Err(cfg_err(format!(
                    "ple_layer_ids={ple_ids:?}: une couche attendue"
                )))
            }
        };
        let gate = text
            .get("output_gate_type")
            .and_then(|v| v.as_str())
            .unwrap_or("silu");
        let eos = match field(text, "eos_token_id")? {
            serde_json::Value::Array(ids) => ids.first().and_then(|v| v.as_u64()),
            value => value.as_u64(),
        }
        .ok_or_else(|| cfg_err("eos_token_id"))?;
        let cfg = Self {
            hidden: usize_of(text, "hidden_size")?,
            layer_kinds: kinds,
            heads: usize_of(text, "num_attention_heads")?,
            kv_heads: usize_of(text, "num_key_value_heads")?,
            head_dim,
            rotary_dim,
            rope_theta,
            linear_key_heads: usize_of(text, "linear_num_key_heads")?,
            linear_value_heads: usize_of(text, "linear_num_value_heads")?,
            linear_key_dim: usize_of(text, "linear_key_head_dim")?,
            linear_value_dim: usize_of(text, "linear_value_head_dim")?,
            conv_kernel: usize_of(text, "linear_conv_kernel_dim")?,
            experts: usize_of(text, "num_experts")?,
            top_k: usize_of(text, "num_experts_per_tok")?,
            moe_inter: usize_of(text, "moe_intermediate_size")?,
            shared_inter: usize_of(text, "shared_expert_intermediate_size")?,
            norm_topk_prob: text
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true),
            hc_count: usize_of(text, "hc_count")?,
            hc_lowrank: usize_of(text, "hc_lowrank")?,
            indexer_budget: usize_of(text, "indexer_budget")?,
            indexer_compress: usize_of(text, "indexer_compress_ratio")?,
            indexer_head_dim: usize_of(text, "indexer_head_dim")?,
            indexer_heads: usize_of(text, "indexer_n_heads")?,
            ple_layer,
            ple_dim: usize_of(text, "ple_embed_dim")?,
            ple_conv_kernel: usize_or(text, "ple_conv_kernel_size", 4)?,
            ngram_size: usize_of(text, "ngram_size")?,
            heads_per_ngram: usize_of(text, "heads_per_ngram")?,
            ngram_vocab_base: usize_of(text, "ngram_vocab_size_base")? as u64,
            vocab: usize_of(text, "vocab_size")?,
            eos,
            rms_eps: text
                .get("rms_norm_eps")
                .and_then(|v| v.as_f64())
                .unwrap_or(1.0e-6) as f32,
            sigmoid_output_gate: match gate {
                "sigmoid" => true,
                "silu" | "swish" => false,
                other => return Err(cfg_err(format!("output_gate_type={other}"))),
            },
            mtp_layers: usize_or(text, "mtp_num_hidden_layers", 0)?,
        };
        cfg.validate(usize_or(text, "indexer_kv_heads", 1)?)?;
        Ok(cfg)
    }

    /// Invariants vérifiés **dans l'ordre** : chaque condition ne s'évalue que
    /// si les précédentes tiennent (aucune division par zéro ni débordement).
    fn validate(&self, indexer_kv_heads: usize) -> Result<()> {
        let fail = |what: &str| Err(cfg_err(format!("invariant violé: {what}")));
        // Bornes : les produits de trois dimensions restent < 2^60.
        let dims = [
            self.hidden,
            self.heads,
            self.kv_heads,
            self.head_dim,
            self.linear_key_heads,
            self.linear_value_heads,
            self.linear_key_dim,
            self.linear_value_dim,
            self.conv_kernel,
            self.experts,
            self.moe_inter,
            self.shared_inter,
            self.hc_count,
            self.hc_lowrank,
            self.indexer_budget,
            self.indexer_compress,
            self.indexer_head_dim,
            self.indexer_heads,
            self.ple_dim,
            self.ple_conv_kernel,
            self.vocab,
        ];
        if dims.iter().any(|d| *d == 0 || *d > MAX_DIM) {
            return fail("dimensions dans 1..=2^20");
        }
        if self.hc_count < 2 {
            return fail("hc_count > 1");
        }
        if indexer_kv_heads != 1 {
            return fail("indexer_kv_heads = 1");
        }
        if self.indexer_budget % self.indexer_compress != 0 {
            return fail("indexer_budget multiple de indexer_compress_ratio");
        }
        if self.layer_kinds[self.ple_layer] != Qwen4ExpLayerKind::Linear {
            return fail("couche PLE de type linear_attention");
        }
        if self.rotary_dim > self.indexer_head_dim || self.rotary_dim > self.head_dim {
            return fail("rotary_dim ≤ indexer_head_dim et head_dim");
        }
        if self.heads % self.kv_heads != 0 {
            return fail("têtes GQA");
        }
        if self.linear_value_heads % self.linear_key_heads != 0 {
            return fail("têtes DeltaNet");
        }
        if self.top_k == 0 || self.top_k > self.experts {
            return fail("1 ≤ top_k ≤ experts");
        }
        if self.ngram_size < 2 || self.ngram_size > 8 || self.heads_per_ngram == 0 {
            return fail("ngram_size dans 2..=8 et heads_per_ngram ≥ 1");
        }
        let heads = (self.ngram_size - 1)
            .checked_mul(self.heads_per_ngram)
            .filter(|h| *h <= MAX_DIM);
        if heads.is_none_or(|h| self.ple_dim % h != 0) {
            return fail("ple_embed_dim divisible par le nombre de têtes n-gram");
        }
        if self.mtp_layers > 1 {
            return fail("mtp_num_hidden_layers ≤ 1");
        }
        Ok(())
    }

    /// Nombre de têtes n-gram (bigrammes puis trigrammes…).
    pub fn ngram_heads(&self) -> usize {
        (self.ngram_size - 1) * self.heads_per_ngram
    }

    /// Largeur du résiduel élargi `hc_count × hidden`.
    pub fn hc_width(&self) -> usize {
        self.hc_count * self.hidden
    }
}

/// Table n-gram E4M3 lue par lignes, échelle per-tensor appliquée une fois.
struct NgramTable {
    path: PathBuf,
    file: File,
    rows: u64,
    width: usize,
    scale: f32,
}

/// Paramètres de hachage n-gram vérifiés.
#[derive(Clone, Debug)]
pub struct Qwen4ExpNgramParams {
    pub multipliers: Vec<i64>,
    pub vocab_sizes: Vec<i64>,
    pub offsets: Vec<i64>,
}

/// Checkpoint `qwen4_exp` q4e8 ouvert et validé.
pub struct Qwen4ExpCheckpoint {
    pub config: Qwen4ExpConfig,
    dir: PathBuf,
    shards: Vec<ShardHeader>,
    files: Vec<File>,
    index: HashMap<String, (usize, ShardTensorEntry)>,
    group_size: usize,
    default_bits: usize,
    overrides: HashMap<String, usize>,
    ngram: NgramTable,
    pub ngram_params: Qwen4ExpNgramParams,
}

fn io(path: &Path, source: std::io::Error) -> InferError {
    InferError::Io {
        path: path.to_path_buf(),
        source,
    }
}

fn read_json(path: &Path) -> Result<serde_json::Value> {
    let raw = std::fs::read(path).map_err(|e| io(path, e))?;
    serde_json::from_slice(&raw).map_err(|source| InferError::Json {
        path: path.to_path_buf(),
        source,
    })
}

/// Rôle attendu d'un nom de tenseur de sortie (sans suffixe pour les affines).
#[derive(Clone, Debug)]
enum Expect {
    /// Affine `[rows, cols]` (ou empilé `[E, rows, cols]` si `stack`).
    Affine {
        rows: usize,
        cols: usize,
        stack: Option<usize>,
    },
    /// Tenseur copié (flottant), forme exacte.
    Float(Vec<usize>),
    /// Métadonnée I64.
    Int(Vec<usize>),
}

fn affine(rows: usize, cols: usize) -> Expect {
    Expect::Affine {
        rows,
        cols,
        stack: None,
    }
}

/// Mixer d'hyper-connections (norme groupée, `W_down`, `W_up`) sous `p`.
fn push_mixer(c: &Qwen4ExpConfig, p: &str, out: &mut Vec<(String, Expect)>) {
    let hc = c.hc_width();
    out.push((format!("{p}hc_norm.weight"), Expect::Float(vec![hc])));
    out.push((
        format!("{p}input_mix_weight_down"),
        affine(c.hc_lowrank, hc),
    ));
    out.push((format!("{p}input_mix_weight_up"), affine(hc, c.hc_lowrank)));
}

/// Tenseurs d'une couche (tronc ou MTP) sous le préfixe `p`.
fn push_layer(
    c: &Qwen4ExpConfig,
    p: &str,
    kind: Qwen4ExpLayerKind,
    with_ple: bool,
    out: &mut Vec<(String, Expect)>,
) {
    let (h, hc) = (c.hidden, c.hc_width());
    for hcn in ["attn_hyper_connection.", "mlp_hyper_connection."] {
        push_mixer(c, &format!("{p}{hcn}"), out);
        out.push((
            format!("{p}{hcn}block_inject_weight.weight"),
            Expect::Float(vec![c.hc_count, hc]),
        ));
    }
    out.push((
        format!("{p}mlp.gate.weight"),
        Expect::Float(vec![c.experts, h]),
    ));
    out.push((
        format!("{p}mlp.shared_expert_gate.weight"),
        Expect::Float(vec![1, h]),
    ));
    out.push((
        format!("{p}mlp.shared_expert.gate_proj"),
        affine(c.shared_inter, h),
    ));
    out.push((
        format!("{p}mlp.shared_expert.up_proj"),
        affine(c.shared_inter, h),
    ));
    out.push((
        format!("{p}mlp.shared_expert.down_proj"),
        affine(h, c.shared_inter),
    ));
    for (proj, rows, cols) in [
        ("gate_proj", c.moe_inter, h),
        ("up_proj", c.moe_inter, h),
        ("down_proj", h, c.moe_inter),
    ] {
        out.push((
            format!("{p}mlp.switch_mlp.{proj}"),
            Expect::Affine {
                rows,
                cols,
                stack: Some(c.experts),
            },
        ));
    }
    let key_dim = c.linear_key_heads * c.linear_key_dim;
    let value_dim = c.linear_value_heads * c.linear_value_dim;
    match kind {
        Qwen4ExpLayerKind::Linear => {
            let l = format!("{p}linear_attn.");
            let vh = c.linear_value_heads;
            out.push((
                format!("{l}in_proj_qkv"),
                affine(2 * key_dim + value_dim, h),
            ));
            out.push((format!("{l}in_proj_z"), affine(value_dim, h)));
            out.push((format!("{l}out_proj"), affine(h, value_dim)));
            out.push((format!("{l}in_proj_a.weight"), Expect::Float(vec![vh, h])));
            out.push((format!("{l}in_proj_b.weight"), Expect::Float(vec![vh, h])));
            out.push((
                format!("{l}conv1d.weight"),
                Expect::Float(vec![2 * key_dim + value_dim, 1, c.conv_kernel]),
            ));
            out.push((format!("{l}A_log"), Expect::Float(vec![vh])));
            out.push((format!("{l}dt_bias"), Expect::Float(vec![vh])));
            out.push((
                format!("{l}norm.weight"),
                Expect::Float(vec![c.linear_value_dim]),
            ));
        }
        Qwen4ExpLayerKind::Full => {
            let a = format!("{p}self_attn.");
            let ihd = c.indexer_head_dim;
            out.push((format!("{a}q_proj"), affine(2 * c.heads * c.head_dim, h)));
            out.push((format!("{a}k_proj"), affine(c.kv_heads * c.head_dim, h)));
            out.push((format!("{a}v_proj"), affine(c.kv_heads * c.head_dim, h)));
            out.push((format!("{a}o_proj"), affine(h, c.heads * c.head_dim)));
            out.push((format!("{a}q_norm.weight"), Expect::Float(vec![c.head_dim])));
            out.push((format!("{a}k_norm.weight"), Expect::Float(vec![c.head_dim])));
            out.push((
                format!("{a}indexer.index_qk_proj"),
                affine((c.indexer_heads + 1) * ihd, h),
            ));
            out.push((
                format!("{a}indexer.q_layernorm.weight"),
                Expect::Float(vec![ihd]),
            ));
            out.push((
                format!("{a}indexer.k_layernorm.weight"),
                Expect::Float(vec![ihd]),
            ));
        }
    }
    if with_ple {
        let e = format!("{p}ple.");
        let heads = c.ngram_heads();
        out.push((format!("{e}key_proj"), affine(hc, c.ple_dim)));
        out.push((format!("{e}value_proj"), affine(h, c.ple_dim)));
        for norm in ["norm_conv", "norm_key", "norm_query"] {
            out.push((format!("{e}{norm}.weight"), Expect::Float(vec![hc])));
        }
        out.push((
            format!("{e}conv1d.weight"),
            Expect::Float(vec![hc, 1, c.ple_conv_kernel]),
        ));
        let pe = format!("{e}ple_embedding.");
        out.push((
            format!("{pe}layer_multipliers"),
            Expect::Int(vec![c.ngram_size]),
        ));
        out.push((
            format!("{pe}ngram_heads_vocab_sizes"),
            Expect::Int(vec![heads]),
        ));
        out.push((format!("{pe}ngram_heads_offsets"), Expect::Int(vec![heads])));
        out.push((
            format!("{pe}ngram_embedding.weight_scale"),
            Expect::Float(vec![1]),
        ));
    }
}

/// Inventaire attendu du tronc (shards indexés).
fn expected_tensors(c: &Qwen4ExpConfig) -> Vec<(String, Expect)> {
    let mut out = vec![
        (format!("{TRUNK}embed_tokens"), affine(c.vocab, c.hidden)),
        (LM_HEAD.to_string(), affine(c.vocab, c.hidden)),
    ];
    push_mixer(c, &format!("{TRUNK}hyper_connection_mixer."), &mut out);
    for (layer, kind) in c.layer_kinds.iter().enumerate() {
        push_layer(
            c,
            &format!("{TRUNK}layers.{layer}."),
            *kind,
            layer == c.ple_layer,
            &mut out,
        );
    }
    out
}

/// Inventaire attendu du sidecar `mtp.safetensors` (une couche QSA + MoE).
fn expected_mtp_tensors(c: &Qwen4ExpConfig) -> Vec<(String, Expect)> {
    let (h, hc) = (c.hidden, c.hc_width());
    let mut out = vec![
        ("mtp.fc_embedding".to_string(), affine(h, h)),
        ("mtp.fc_hidden".to_string(), affine(h, h)),
        (
            "mtp.pre_fc_norm_embedding.weight".to_string(),
            Expect::Float(vec![h]),
        ),
        (
            "mtp.pre_fc_norm_hidden.weight".to_string(),
            Expect::Float(vec![hc]),
        ),
    ];
    push_mixer(c, "mtp.hyper_connection_mixer.", &mut out);
    push_layer(c, "mtp.layers.0.", Qwen4ExpLayerKind::Full, false, &mut out);
    out
}

/// Indexe les tenseurs d'un shard après contrôle physique : dtype connu,
/// longueur = produit(forme) × taille, plage dans le fichier.
fn index_shard(
    header: &ShardHeader,
    file_len: u64,
    shard: usize,
    entries: &mut HashMap<String, (usize, ShardTensorEntry)>,
) -> Result<()> {
    for (tensor, entry) in &header.tensors {
        let size = dtype_size(entry.dtype)
            .ok_or_else(|| cfg_err(format!("{tensor}: dtype {:?} inattendu", entry.dtype)))?;
        let expected = entry
            .shape
            .iter()
            .try_fold(size, |acc, d| acc.checked_mul(*d))
            .ok_or_else(|| cfg_err(format!("{tensor}: forme trop grande")))?;
        let [start, end] = entry.data_offsets;
        let physical = header.data_start.checked_add(end as u64);
        if end.checked_sub(start) != Some(expected) || physical.is_none_or(|p| p > file_len) {
            return Err(cfg_err(format!("{tensor}: plage incohérente")));
        }
        if entries
            .insert(tensor.clone(), (shard, entry.clone()))
            .is_some()
        {
            return Err(cfg_err(format!("{tensor}: doublon")));
        }
    }
    Ok(())
}

fn dtype_size(dtype: Dtype) -> Option<usize> {
    match dtype {
        Dtype::U8 | Dtype::F8_E4M3 => Some(1),
        Dtype::BF16 | Dtype::F16 => Some(2),
        Dtype::F32 | Dtype::U32 => Some(4),
        Dtype::I64 => Some(8),
        _ => None,
    }
}

impl Qwen4ExpCheckpoint {
    /// Ouvre et valide un dossier produit par `convert-qwen4exp`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur sur config invalide, tenseur manquant, inattendu ou
    /// mal formé, bits incohérents avec la config, plage hors fichier, table
    /// n-gram ou paramètres de hachage incohérents.
    pub fn open(dir: &Path) -> Result<Self> {
        let config_json = read_json(&dir.join("config.json"))?;
        let config = Qwen4ExpConfig::from_json(&config_json)?;
        let quant = config_json
            .get("quantization")
            .and_then(|q| q.as_object())
            .ok_or_else(|| cfg_err("bloc quantization absent"))?;
        let group_size = quant
            .get("group_size")
            .and_then(|v| v.as_u64())
            .filter(|gs| *gs > 0 && *gs as usize <= MAX_DIM)
            .ok_or_else(|| cfg_err("quantization.group_size dans 1..=2^20"))?
            as usize;
        let default_bits = quant
            .get("bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| cfg_err("quantization.bits"))? as usize;
        let valid_bits = |bits: u64| matches!(bits, 2 | 3 | 4 | 5 | 6 | 8);
        if !valid_bits(default_bits as u64) {
            return Err(cfg_err(format!(
                "quantization.bits={default_bits} hors {{2,3,4,5,6,8}}"
            )));
        }
        let mut overrides = HashMap::new();
        for (key, value) in quant {
            if let Some(bits) = value.get("bits").and_then(|v| v.as_u64()) {
                if !valid_bits(bits) {
                    return Err(cfg_err(format!("{key}: bits={bits} hors {{2,3,4,5,6,8}}")));
                }
                let gs = value.get("group_size").and_then(|v| v.as_u64());
                if gs != Some(group_size as u64) {
                    return Err(cfg_err(format!("{key}: group_size {gs:?} ≠ {group_size}")));
                }
                overrides.insert(key.clone(), bits as usize);
            }
        }
        let index = read_json(&dir.join("model.safetensors.index.json"))?;
        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| cfg_err("index sans weight_map"))?;
        let names: BTreeSet<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect();
        let mut shards = Vec::new();
        let mut files = Vec::new();
        let mut entries = HashMap::new();
        for (shard_index, name) in names.iter().enumerate() {
            let path = dir.join(name);
            let header = read_shard_header(&path)?;
            let file = File::open(&path).map_err(|e| io(&path, e))?;
            let len = file.metadata().map_err(|e| io(&path, e))?.len();
            for tensor in header.tensors.keys() {
                if weight_map.get(tensor).and_then(|v| v.as_str()) != Some(name.as_str()) {
                    return Err(cfg_err(format!("{tensor}: index et en-tête divergent")));
                }
            }
            index_shard(&header, len, shard_index, &mut entries)?;
            shards.push(header);
            files.push(file);
        }
        if entries.len() != weight_map.len() {
            return Err(cfg_err("index et en-têtes de tailles différentes"));
        }
        let ngram = Self::open_ngram(dir, &config)?;
        let mut checkpoint = Self {
            config,
            dir: dir.to_path_buf(),
            shards,
            files,
            index: entries,
            group_size,
            default_bits,
            overrides,
            ngram,
            ngram_params: Qwen4ExpNgramParams {
                multipliers: Vec::new(),
                vocab_sizes: Vec::new(),
                offsets: Vec::new(),
            },
        };
        checkpoint.validate_inventory()?;
        checkpoint.ngram_params = checkpoint.verify_ngram_params()?;
        if checkpoint.config.mtp_layers == 1 {
            checkpoint.validate_mtp_metadata()?;
        }
        Ok(checkpoint)
    }

    fn open_ngram(dir: &Path, config: &Qwen4ExpConfig) -> Result<NgramTable> {
        let meta = read_json(&dir.join("ngram/meta.json"))?;
        let shape = meta
            .get("shape")
            .and_then(|v| v.as_array())
            .and_then(|s| Some((s.first()?.as_u64()?, s.get(1)?.as_u64()?)))
            .ok_or_else(|| cfg_err("ngram/meta.json: shape"))?;
        let scale = meta
            .get("weight_scale")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| cfg_err("ngram/meta.json: weight_scale"))? as f32;
        let file_name = meta
            .get("file")
            .and_then(|v| v.as_str())
            .ok_or_else(|| cfg_err("ngram/meta.json: file"))?;
        let width = usize::try_from(shape.1)
            .ok()
            .filter(|w| *w <= MAX_DIM)
            .ok_or_else(|| cfg_err(format!("ngram/meta.json: largeur {}", shape.1)))?;
        if width.checked_mul(config.ngram_heads()) != Some(config.ple_dim)
            || !scale.is_finite()
            || scale <= 0.0
        {
            return Err(cfg_err(format!(
                "table n-gram largeur {width}, échelle {scale} incohérentes"
            )));
        }
        let expected_len = shape
            .0
            .checked_mul(shape.1)
            .ok_or_else(|| cfg_err("ngram/meta.json: forme trop grande"))?;
        let path = ngram_table_path(dir, file_name)?;
        let file = File::open(&path).map_err(|e| io(&path, e))?;
        let len = file.metadata().map_err(|e| io(&path, e))?.len();
        if len != expected_len {
            return Err(cfg_err(format!(
                "table n-gram: {len} octets, {expected_len} attendus"
            )));
        }
        Ok(NgramTable {
            path,
            file,
            rows: shape.0,
            width,
            scale,
        })
    }

    fn validate_inventory(&self) -> Result<()> {
        self.validate_entries(&self.index, &expected_tensors(&self.config))
    }

    /// Vérifie que `index` contient exactement `expected` (formes, dtypes, triplets).
    fn validate_entries(
        &self,
        index: &HashMap<String, (usize, ShardTensorEntry)>,
        expected: &[(String, Expect)],
    ) -> Result<()> {
        let entry = |name: &str| {
            index
                .get(name)
                .map(|(_, e)| e)
                .ok_or_else(|| InferError::MissingWeight(name.to_string()))
        };
        let expect_entry = |name: &str, dtype: Dtype, shape: &[usize]| -> Result<()> {
            let e = entry(name)?;
            if e.dtype != dtype || e.shape != shape {
                return Err(cfg_err(format!(
                    "{name}: {:?} {:?}, attendu {dtype:?} {shape:?}",
                    e.dtype, e.shape
                )));
            }
            Ok(())
        };
        let mut seen = BTreeSet::new();
        for (name, expect) in expected.iter().cloned() {
            match expect {
                Expect::Affine { rows, cols, stack } => {
                    let bits = self.bits_of(&name);
                    let groups = cols / self.group_size;
                    // bits ∈ {2,3,4,5,6,8} (vérifié à l'ouverture) : produit borné.
                    let packed_bits = cols
                        .checked_mul(bits)
                        .ok_or_else(|| cfg_err(format!("{name}: {cols}×{bits} déborde")))?;
                    if cols % self.group_size != 0 || packed_bits % 32 != 0 {
                        return Err(cfg_err(format!("{name}: {cols} colonnes / u{bits}")));
                    }
                    let lead: Vec<usize> = stack.into_iter().collect();
                    let shape = |last| {
                        let mut s = lead.clone();
                        s.extend([rows, last]);
                        s
                    };
                    expect_entry(
                        &format!("{name}.weight"),
                        Dtype::U32,
                        &shape(packed_bits / 32),
                    )?;
                    expect_entry(&format!("{name}.scales"), Dtype::BF16, &shape(groups))?;
                    expect_entry(&format!("{name}.biases"), Dtype::BF16, &shape(groups))?;
                    for suffix in [".weight", ".scales", ".biases"] {
                        seen.insert(format!("{name}{suffix}"));
                    }
                }
                Expect::Float(shape) => {
                    let entry = entry(&name)?;
                    if !matches!(entry.dtype, Dtype::BF16 | Dtype::F32 | Dtype::F16)
                        || entry.shape != shape
                    {
                        return Err(cfg_err(format!(
                            "{name}: {:?} {:?}, attendu flottant {shape:?}",
                            entry.dtype, entry.shape
                        )));
                    }
                    seen.insert(name);
                }
                Expect::Int(shape) => {
                    expect_entry(&name, Dtype::I64, &shape)?;
                    seen.insert(name);
                }
            }
        }
        if let Some(extra) = index.keys().find(|name| !seen.contains(*name)) {
            return Err(cfg_err(format!("tenseur inattendu: {extra}")));
        }
        Ok(())
    }

    fn verify_ngram_params(&self) -> Result<Qwen4ExpNgramParams> {
        let c = &self.config;
        let p = format!("{TRUNK}layers.{}.ple.ple_embedding.", c.ple_layer);
        let multipliers = self.i64s(&format!("{p}layer_multipliers"))?;
        let vocab_sizes = self.i64s(&format!("{p}ngram_heads_vocab_sizes"))?;
        let offsets = self.i64s(&format!("{p}ngram_heads_offsets"))?;
        if multipliers != layer_multipliers(c.vocab as u64, c.ngram_size, 0, NGRAM_SEED)
            || vocab_sizes != head_vocab_sizes(c.ngram_vocab_base, c.ngram_heads())
        {
            return Err(cfg_err(
                "buffers de hachage n-gram ≠ recalcul SplitMix64 (seed 1234)",
            ));
        }
        let mut running = 0_i64;
        for (offset, size) in offsets.iter().zip(&vocab_sizes) {
            if *offset != running {
                return Err(cfg_err("ngram_heads_offsets incohérents"));
            }
            running += size;
        }
        if running as u64 > self.ngram.rows {
            return Err(cfg_err(
                "table n-gram plus courte que le vocabulaire des têtes",
            ));
        }
        let scale = self.floats(&format!("{p}ngram_embedding.weight_scale"))?;
        if scale.first().copied() != Some(self.ngram.scale) {
            return Err(cfg_err("weight_scale du shard ≠ ngram/meta.json"));
        }
        Ok(Qwen4ExpNgramParams {
            multipliers,
            vocab_sizes,
            offsets,
        })
    }

    /// Inventaire complet du sidecar MTP sur métadonnées seules (formes,
    /// dtypes, triplets, plages physiques), sans lire les valeurs.
    fn validate_mtp_metadata(&self) -> Result<()> {
        let path = self.dir.join("mtp.safetensors");
        let header = read_shard_header(&path)?;
        let len = std::fs::metadata(&path).map_err(|e| io(&path, e))?.len();
        let mut index = HashMap::new();
        index_shard(&header, len, 0, &mut index)?;
        self.validate_entries(&index, &expected_mtp_tensors(&self.config))
            .map_err(|e| cfg_err(format!("MTP: {e}")))
    }

    fn entry(&self, name: &str) -> Result<(usize, &ShardTensorEntry)> {
        self.index
            .get(name)
            .map(|(shard, entry)| (*shard, entry))
            .ok_or_else(|| InferError::MissingWeight(name.to_string()))
    }

    fn read(&self, name: &str, start: usize, len: usize) -> Result<Vec<u8>> {
        let (shard, entry) = self.entry(name)?;
        let total = entry.data_offsets[1] - entry.data_offsets[0];
        if start.checked_add(len).is_none_or(|end| end > total) {
            return Err(InferError::Shape(format!("{name}: lecture hors tenseur")));
        }
        let offset = self.shards[shard].data_start + (entry.data_offsets[0] + start) as u64;
        let mut buf = vec![0_u8; len];
        self.files[shard]
            .read_exact_at(&mut buf, offset)
            .map_err(|e| io(&self.shards[shard].path, e))?;
        Ok(buf)
    }

    fn bits_of(&self, base: &str) -> usize {
        self.overrides
            .get(base)
            .copied()
            .unwrap_or(self.default_bits)
    }

    /// Tenseur flottant copié (BF16/F32), converti en f32.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tenseur manque ou n'est pas flottant.
    pub fn floats(&self, name: &str) -> Result<Vec<f32>> {
        let (_, entry) = self.entry(name)?;
        let len = entry.data_offsets[1] - entry.data_offsets[0];
        bytes_to_dense_f32(&self.read(name, 0, len)?, entry.dtype, name)
    }

    /// Métadonnée I64.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tenseur manque ou n'est pas I64.
    pub fn i64s(&self, name: &str) -> Result<Vec<i64>> {
        let (_, entry) = self.entry(name)?;
        if entry.dtype != Dtype::I64 {
            return Err(cfg_err(format!("{name}: I64 attendu")));
        }
        let len = entry.data_offsets[1] - entry.data_offsets[0];
        Ok(self
            .read(name, 0, len)?
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect())
    }

    /// Lignes `rows` d'un affine (éventuellement l'expert `stack` d'une pile),
    /// déquantifiées en f32 `[rows.len(), cols]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'affine manque ou si la plage sort du tenseur.
    pub fn affine_rows(
        &self,
        base: &str,
        stack: Option<usize>,
        rows: Range<usize>,
    ) -> Result<Tensor> {
        let (_, weight) = self.entry(&format!("{base}.weight"))?;
        let (_, scales) = self.entry(&format!("{base}.scales"))?;
        let rank = weight.shape.len();
        let (total_rows, packed) = (weight.shape[rank - 2], weight.shape[rank - 1]);
        let groups = scales.shape[rank - 1];
        let bits = self.bits_of(base);
        if rows.end > total_rows || rows.is_empty() {
            return Err(InferError::Shape(format!(
                "{base}: lignes {rows:?} hors [0, {total_rows})"
            )));
        }
        let row0 = stack.unwrap_or(0) * total_rows + rows.start;
        let n = rows.len();
        let words = self.read(&format!("{base}.weight"), row0 * packed * 4, n * packed * 4)?;
        let words: Vec<u32> = words
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let s = self.read(&format!("{base}.scales"), row0 * groups * 2, n * groups * 2)?;
        let b = self.read(&format!("{base}.biases"), row0 * groups * 2, n * groups * 2)?;
        let s = Tensor::from_vec(vec![n, groups], bytes_to_dense_f32(&s, Dtype::BF16, base)?)?;
        let b = Tensor::from_vec(vec![n, groups], bytes_to_dense_f32(&b, Dtype::BF16, base)?)?;
        dequantize_affine_u32(&[n, packed], &words, &s, &b, self.group_size, bits)
    }

    /// Triplet affine brut (packed, échelles et biais bf16), sans
    /// déquantification : un expert de pile `switch_mlp` si `stack` est donné.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le tenseur manque, si l'expert sort de la pile,
    /// ou sur lecture.
    pub fn quantized(&self, base: &str, stack: Option<usize>) -> Result<AffineQuantizedTensor> {
        let (_, weight) = self.entry(&format!("{base}.weight"))?;
        let (_, scales) = self.entry(&format!("{base}.scales"))?;
        let rank = weight.shape.len();
        let stacked = rank == 3;
        if stacked != stack.is_some() || stack.is_some_and(|e| e >= weight.shape[0]) {
            return Err(InferError::Shape(format!(
                "{base}: expert {stack:?} hors pile"
            )));
        }
        let (rows, packed) = (weight.shape[rank - 2], weight.shape[rank - 1]);
        let groups = scales.shape[rank - 1];
        let row0 = stack.unwrap_or(0) * rows;
        let words: Vec<u32> = self
            .read(
                &format!("{base}.weight"),
                row0 * packed * 4,
                rows * packed * 4,
            )?
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let s = self.read(
            &format!("{base}.scales"),
            row0 * groups * 2,
            rows * groups * 2,
        )?;
        let b = self.read(
            &format!("{base}.biases"),
            row0 * groups * 2,
            rows * groups * 2,
        )?;
        AffineQuantizedTensor::new(
            &[rows, packed],
            words,
            Tensor::from_vec(
                vec![rows, groups],
                bytes_to_dense_f32(&s, Dtype::BF16, base)?,
            )?,
            Tensor::from_vec(
                vec![rows, groups],
                bytes_to_dense_f32(&b, Dtype::BF16, base)?,
            )?,
            self.group_size,
            self.bits_of(base),
        )
    }

    /// Affine complet déquantifié en f32 `[rows, cols]`.
    ///
    /// # Errors
    ///
    /// Voir [`Self::affine_rows`].
    pub fn affine(&self, base: &str) -> Result<Tensor> {
        let (_, weight) = self.entry(&format!("{base}.weight"))?;
        let rows = weight.shape[weight.shape.len() - 2];
        self.affine_rows(base, None, 0..rows)
    }

    /// Expert `index` d'une pile `switch_mlp` déquantifié en f32.
    ///
    /// # Errors
    ///
    /// Voir [`Self::affine_rows`].
    pub fn expert(&self, base: &str, index: usize) -> Result<Tensor> {
        let (_, weight) = self.entry(&format!("{base}.weight"))?;
        if weight.shape.len() != 3 || index >= weight.shape[0] {
            return Err(InferError::Shape(format!(
                "{base}: expert {index} hors pile"
            )));
        }
        self.affine_rows(base, Some(index), 0..weight.shape[1])
    }

    /// Lignes de la table n-gram `ids`, `[ids.len(), width]`, échelle appliquée.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un identifiant sort de la table, sur lecture
    /// tronquée ou octet NaN.
    pub fn ngram_rows(&self, ids: &[u64]) -> Result<Vec<f32>> {
        let width = self.ngram.width;
        let mut out = vec![0.0_f32; ids.len() * width];
        let mut row = vec![0_u8; width];
        for (slot, &id) in out.chunks_exact_mut(width).zip(ids) {
            if id >= self.ngram.rows {
                return Err(InferError::Shape(format!("ligne n-gram {id} hors table")));
            }
            self.ngram
                .file
                .read_exact_at(&mut row, id * width as u64)
                .map_err(|e| io(&self.ngram.path, e))?;
            for (value, &byte) in slot.iter_mut().zip(&row) {
                *value = f8_e4m3_to_f32(byte)
                    .ok_or_else(|| cfg_err(format!("ligne n-gram {id}: NaN")))?
                    * self.ngram.scale;
            }
        }
        Ok(out)
    }

    /// Octets stockés des tenseurs dont le nom commence par `prefix`, par
    /// nature : `(packed, échelles et biais, flottants copiés)`.
    pub fn tensor_bytes_by_kind(&self, prefix: &str) -> (u64, u64, u64) {
        let mut sizes = (0_u64, 0_u64, 0_u64);
        for (name, (_, entry)) in self
            .index
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
        {
            let bytes = (entry.data_offsets[1] - entry.data_offsets[0]) as u64;
            if name.ends_with(".scales") || name.ends_with(".biases") {
                sizes.1 += bytes;
            } else if entry.dtype == Dtype::U32 {
                sizes.0 += bytes;
            } else {
                sizes.2 += bytes;
            }
        }
        sizes
    }

    /// Largeur d'une ligne n-gram (octets E4M3).
    pub fn ngram_width(&self) -> usize {
        self.ngram.width
    }

    /// Valeurs des 256 octets E4M3 multipliées par l'échelle per-tensor ; NaN
    /// pour les deux codes NaN (refusés à la lecture).
    pub fn ngram_lut(&self) -> Vec<f32> {
        (0..=255_u8)
            .map(|byte| f8_e4m3_to_f32(byte).map_or(f32::NAN, |v| v * self.ngram.scale))
            .collect()
    }

    /// Lit les lignes `ids` brutes (E4M3) dans `out` (`ids.len()·largeur`
    /// octets), une lecture positionnelle par ligne, en parallèle. Refuse une
    /// ligne hors table ou contenant un NaN, comme [`Self::ngram_rows`].
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de taille, d'index ou de NaN.
    pub fn ngram_rows_raw(&self, ids: &[u64], out: &mut [u8]) -> Result<()> {
        use rayon::prelude::*;
        let width = self.ngram.width;
        if ids.len().checked_mul(width) != Some(out.len()) {
            return Err(InferError::Shape(format!(
                "lignes n-gram : {} octets pour {} lignes de {width}",
                out.len(),
                ids.len()
            )));
        }
        out.par_chunks_mut(width)
            .zip(ids.par_iter())
            .try_for_each(|(slot, &id)| {
                if id >= self.ngram.rows {
                    return Err(InferError::Shape(format!("ligne n-gram {id} hors table")));
                }
                self.ngram
                    .file
                    .read_exact_at(slot, id * width as u64)
                    .map_err(|e| io(&self.ngram.path, e))?;
                if slot.iter().any(|b| b & 0x7f == 0x7f) {
                    return Err(cfg_err(format!("ligne n-gram {id}: NaN")));
                }
                Ok(())
            })
    }

    /// Préfixe d'une couche du tronc en sortie.
    pub fn layer_prefix(layer: usize) -> String {
        format!("{TRUNK}layers.{layer}.")
    }

    /// Nom de base d'un tenseur global du tronc.
    pub fn trunk(name: &str) -> String {
        format!("{TRUNK}{name}")
    }

    /// Nom de base du lm_head.
    pub fn lm_head() -> &'static str {
        LM_HEAD
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mini_config() -> serde_json::Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/qwen4exp_mini/source/config.json");
        read_json(&path).expect("invariant: config de fixture")
    }

    fn with_text(edit: impl FnOnce(&mut serde_json::Value)) -> Result<Qwen4ExpConfig> {
        let mut config = mini_config();
        edit(&mut config["text_config"]);
        Qwen4ExpConfig::from_json(&config)
    }

    #[test]
    fn degenerate_ngram_configs_are_errors_not_panics() {
        for (key, value) in [
            ("ngram_size", 1),
            ("ngram_size", 0),
            ("heads_per_ngram", 0),
            ("indexer_compress_ratio", 0),
            ("num_key_value_heads", 0),
            ("linear_num_key_heads", 0),
            ("hidden_size", 0),
            ("hc_count", 1),
        ] {
            let result = with_text(|t| t[key] = value.into());
            assert!(result.is_err(), "{key}={value} accepté");
        }
        assert!(with_text(|t| t["ple_embed_dim"] = 63.into()).is_err());
        assert!(with_text(|t| t["hidden_size"] = (1_u64 << 40).into()).is_err());
    }

    #[test]
    fn both_output_gates_are_accepted() {
        let sigmoid = with_text(|_| {}).expect("invariant: config fixture");
        assert!(sigmoid.sigmoid_output_gate);
        let silu = with_text(|t| t["output_gate_type"] = "silu".into()).expect("invariant: silu");
        assert!(!silu.sigmoid_output_gate);
        assert!(with_text(|t| t["output_gate_type"] = "relu".into()).is_err());
    }

    #[cfg(feature = "devtools")]
    #[test]
    fn ngram_raw_rows_with_lut_match_dequantized_rows() {
        let dst = converted_fixture();
        let ckpt = Qwen4ExpCheckpoint::open(dst.path()).expect("invariant: fixture ouverte");
        let width = ckpt.ngram_width();
        let ids = [0_u64, 5, 3, 0, 17];
        let mut raw = vec![0_u8; ids.len() * width];
        ckpt.ngram_rows_raw(&ids, &mut raw)
            .expect("invariant: lecture brute");
        let lut = ckpt.ngram_lut();
        let decoded: Vec<f32> = raw.iter().map(|&b| lut[usize::from(b)]).collect();
        assert_eq!(decoded, ckpt.ngram_rows(&ids).expect("invariant: lignes"));
        let mut one = vec![0_u8; width];
        assert!(ckpt.ngram_rows_raw(&[u64::MAX], &mut one).is_err());
        assert!(ckpt.ngram_rows_raw(&ids, &mut one).is_err());
    }

    #[cfg(feature = "devtools")]
    #[test]
    fn raw_quantized_triplets_dequantize_like_affine_reads() {
        let dst = converted_fixture();
        let ckpt = Qwen4ExpCheckpoint::open(dst.path()).expect("invariant: fixture ouverte");
        let dense = Qwen4ExpCheckpoint::layer_prefix(0) + "mlp.shared_expert.down_proj";
        let raw = ckpt.quantized(&dense, None).expect("invariant: triplet");
        assert_eq!(
            raw.dequantize().expect("invariant: déquant").data(),
            ckpt.affine(&dense).expect("invariant: affine").data()
        );
        let stack = Qwen4ExpCheckpoint::layer_prefix(0) + "mlp.switch_mlp.down_proj";
        let raw = ckpt.quantized(&stack, Some(3)).expect("invariant: expert");
        assert_eq!(
            raw.dequantize().expect("invariant: déquant").data(),
            ckpt.expert(&stack, 3).expect("invariant: expert").data()
        );
        assert!(ckpt.quantized(&stack, None).is_err());
        assert!(ckpt.quantized(&stack, Some(ckpt.config.experts)).is_err());
        assert!(ckpt.quantized(&dense, Some(0)).is_err());
    }

    /// Convertit la fixture N1 dans un dossier temporaire.
    #[cfg(feature = "devtools")]
    fn converted_fixture() -> tempfile::TempDir {
        use crate::devtools::convert::{convert_qwen4exp, ConvertOptions};
        let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
        convert_qwen4exp(&ConvertOptions {
            src: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4exp_mini/source"),
            dst: dst.path().to_path_buf(),
            source_revision: "fixture".to_string(),
            source_repo: None,
            converter_commit: None,
            only: None,
            verify: false,
            plan_only: false,
        })
        .expect("invariant: conversion");
        dst
    }

    #[cfg(feature = "devtools")]
    fn edit_json(path: &Path, edit: impl FnOnce(&mut serde_json::Value)) {
        let mut value = read_json(path).expect("invariant: json");
        edit(&mut value);
        std::fs::write(path, value.to_string()).expect("invariant: écriture");
    }

    #[cfg(feature = "devtools")]
    #[test]
    fn hostile_quantization_and_ngram_metadata_are_errors_not_panics() {
        type Edit = fn(&mut serde_json::Value);
        let config_cases: [(Edit, &str); 4] = [
            (|c| c["quantization"]["group_size"] = 0.into(), "group_size"),
            (|c| c["quantization"]["bits"] = (1_u64 << 63).into(), "bits"),
            (|c| c["quantization"]["bits"] = 7.into(), "bits"),
            (
                |c| c["quantization"]["language_model.lm_head"]["bits"] = (1_u64 << 62).into(),
                "bits",
            ),
        ];
        for (edit, needle) in config_cases {
            let dst = converted_fixture();
            edit_json(&dst.path().join("config.json"), edit);
            let error = Qwen4ExpCheckpoint::open(dst.path())
                .err()
                .unwrap_or_else(|| panic!("{needle}: config hostile acceptée"));
            assert!(error.to_string().contains(needle), "{error}");
        }
        let meta_cases: [Edit; 6] = [
            |m| m["shape"] = serde_json::json!([u64::MAX, u64::MAX]),
            |m| m["shape"] = serde_json::json!([4, u64::MAX]),
            |m| m["shape"][1] = 17.into(),
            |m| m["file"] = "/etc/hosts".into(),
            |m| m["file"] = "../config.json".into(),
            |m| m["file"] = "".into(),
        ];
        for edit in meta_cases {
            let dst = converted_fixture();
            edit_json(&dst.path().join("ngram/meta.json"), edit);
            assert!(
                Qwen4ExpCheckpoint::open(dst.path()).is_err(),
                "meta hostile acceptée"
            );
        }
    }

    /// Écrit un safetensors minimal (données nulles) depuis un inventaire.
    #[cfg(feature = "devtools")]
    fn write_inventory(path: &Path, expected: &[(String, Expect)], group_size: usize, skip: &str) {
        let mut header = serde_json::Map::new();
        let mut offset = 0_usize;
        let mut push = |name: String, dtype: &str, shape: Vec<usize>, size: usize| {
            if name == skip {
                return;
            }
            let bytes = shape.iter().product::<usize>() * size;
            header.insert(
                name,
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + bytes]}),
            );
            offset += bytes;
        };
        for (name, expect) in expected {
            match expect {
                Expect::Affine { rows, cols, stack } => {
                    let lead: Vec<usize> = stack.iter().copied().collect();
                    let shape = |last: usize| {
                        let mut s = lead.clone();
                        s.extend([*rows, last]);
                        s
                    };
                    push(format!("{name}.weight"), "U32", shape(cols * 4 / 32), 4);
                    push(
                        format!("{name}.scales"),
                        "BF16",
                        shape(cols / group_size),
                        2,
                    );
                    push(
                        format!("{name}.biases"),
                        "BF16",
                        shape(cols / group_size),
                        2,
                    );
                }
                Expect::Float(shape) => push(name.clone(), "BF16", shape.clone(), 2),
                Expect::Int(shape) => push(name.clone(), "I64", shape.clone(), 8),
            }
        }
        let mut json = serde_json::to_vec(&header).expect("invariant: json");
        while json.len() % 8 != 0 {
            json.push(b' ');
        }
        let mut out = (json.len() as u64).to_le_bytes().to_vec();
        out.extend(json);
        out.resize(out.len() + offset, 0);
        std::fs::write(path, out).expect("invariant: écriture");
    }

    #[cfg(feature = "devtools")]
    #[test]
    fn mtp_sidecar_inventory_is_fully_validated() {
        use crate::devtools::convert::{convert_qwen4exp, ConvertOptions};
        let dst = tempfile::tempdir().expect("invariant: dossier temporaire");
        convert_qwen4exp(&ConvertOptions {
            src: Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen4exp_mini/source"),
            dst: dst.path().to_path_buf(),
            source_revision: "fixture".to_string(),
            source_repo: None,
            converter_commit: None,
            only: None,
            verify: false,
            plan_only: false,
        })
        .expect("invariant: conversion");
        let config_path = dst.path().join("config.json");
        let mut config = read_json(&config_path).expect("invariant: config");
        config["text_config"]["mtp_num_hidden_layers"] = 1.into();
        std::fs::write(&config_path, config.to_string()).expect("invariant: écriture");
        let parsed = Qwen4ExpConfig::from_json(&config).expect("invariant: config MTP");
        let mtp = dst.path().join("mtp.safetensors");
        let expected = expected_mtp_tensors(&parsed);
        write_inventory(&mtp, &expected, 64, "");
        Qwen4ExpCheckpoint::open(dst.path()).expect("invariant: MTP complet accepté");

        write_inventory(&mtp, &expected, 64, "mtp.layers.0.self_attn.o_proj.scales");
        let error = Qwen4ExpCheckpoint::open(dst.path())
            .err()
            .map(|e| e.to_string());
        assert!(
            error.as_deref().is_some_and(|e| e.contains("MTP")),
            "{error:?}"
        );

        write_inventory(&mtp, &expected, 64, "");
        let bytes = std::fs::read(&mtp).expect("invariant: lecture");
        std::fs::write(&mtp, &bytes[..bytes.len() - 16]).expect("invariant: troncature");
        assert!(
            Qwen4ExpCheckpoint::open(dst.path()).is_err(),
            "MTP tronqué accepté"
        );
    }
}
