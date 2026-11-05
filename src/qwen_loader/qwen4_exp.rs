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
use crate::{dequantize_affine_u32, InferError, Result, Tensor};

const TRUNK: &str = "language_model.model.";
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
    pub sigmoid_output_gate: bool,
    pub mtp_layers: usize,
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

    fn validate(&self, indexer_kv_heads: usize) -> Result<()> {
        let checks = [
            (self.hc_count > 1, "hc_count > 1"),
            (indexer_kv_heads == 1, "indexer_kv_heads = 1"),
            (
                self.indexer_compress > 0 && self.indexer_budget % self.indexer_compress == 0,
                "indexer_budget multiple de indexer_compress_ratio",
            ),
            (
                self.layer_kinds[self.ple_layer] == Qwen4ExpLayerKind::Linear,
                "couche PLE de type linear_attention",
            ),
            (
                self.rotary_dim <= self.indexer_head_dim && self.rotary_dim <= self.head_dim,
                "rotary_dim ≤ indexer_head_dim et head_dim",
            ),
            (
                self.heads > 0 && self.kv_heads > 0 && self.heads % self.kv_heads == 0,
                "têtes GQA",
            ),
            (
                self.linear_key_heads > 0 && self.linear_value_heads % self.linear_key_heads == 0,
                "têtes DeltaNet",
            ),
            (
                self.top_k >= 1 && self.top_k <= self.experts,
                "top_k ≤ experts",
            ),
            (self.ngram_size >= 2 && self.heads_per_ngram >= 1, "n-gram"),
            (
                self.ple_dim % ((self.ngram_size - 1) * self.heads_per_ngram) == 0,
                "ple_embed_dim divisible par le nombre de têtes n-gram",
            ),
            (self.mtp_layers <= 1, "mtp_num_hidden_layers ≤ 1"),
        ];
        for (ok, what) in checks {
            if !ok {
                return Err(cfg_err(format!("invariant violé: {what}")));
            }
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

fn expected_tensors(c: &Qwen4ExpConfig) -> Vec<(String, Expect)> {
    let (h, hc) = (c.hidden, c.hc_width());
    let affine = |rows, cols| Expect::Affine {
        rows,
        cols,
        stack: None,
    };
    let mut out: Vec<(String, Expect)> = Vec::new();
    let mut add = |name: String, e: Expect| out.push((name, e));
    add(format!("{TRUNK}embed_tokens"), affine(c.vocab, h));
    add(LM_HEAD.to_string(), affine(c.vocab, h));
    let mixer = |add: &mut dyn FnMut(String, Expect), p: &str| {
        add(format!("{p}hc_norm.weight"), Expect::Float(vec![hc]));
        add(
            format!("{p}input_mix_weight_down"),
            affine(c.hc_lowrank, hc),
        );
        add(format!("{p}input_mix_weight_up"), affine(hc, c.hc_lowrank));
    };
    mixer(&mut add, &format!("{TRUNK}hyper_connection_mixer."));
    let key_dim = c.linear_key_heads * c.linear_key_dim;
    let value_dim = c.linear_value_heads * c.linear_value_dim;
    for (layer, kind) in c.layer_kinds.iter().enumerate() {
        let p = format!("{TRUNK}layers.{layer}.");
        for hcn in ["attn_hyper_connection.", "mlp_hyper_connection."] {
            mixer(&mut add, &format!("{p}{hcn}"));
            add(
                format!("{p}{hcn}block_inject_weight.weight"),
                Expect::Float(vec![c.hc_count, hc]),
            );
        }
        add(
            format!("{p}mlp.gate.weight"),
            Expect::Float(vec![c.experts, h]),
        );
        add(
            format!("{p}mlp.shared_expert_gate.weight"),
            Expect::Float(vec![1, h]),
        );
        add(
            format!("{p}mlp.shared_expert.gate_proj"),
            affine(c.shared_inter, h),
        );
        add(
            format!("{p}mlp.shared_expert.up_proj"),
            affine(c.shared_inter, h),
        );
        add(
            format!("{p}mlp.shared_expert.down_proj"),
            affine(h, c.shared_inter),
        );
        for (proj, rows, cols) in [
            ("gate_proj", c.moe_inter, h),
            ("up_proj", c.moe_inter, h),
            ("down_proj", h, c.moe_inter),
        ] {
            add(
                format!("{p}mlp.switch_mlp.{proj}"),
                Expect::Affine {
                    rows,
                    cols,
                    stack: Some(c.experts),
                },
            );
        }
        match kind {
            Qwen4ExpLayerKind::Linear => {
                let l = format!("{p}linear_attn.");
                add(
                    format!("{l}in_proj_qkv"),
                    affine(2 * key_dim + value_dim, h),
                );
                add(format!("{l}in_proj_z"), affine(value_dim, h));
                add(format!("{l}out_proj"), affine(h, value_dim));
                add(
                    format!("{l}in_proj_a.weight"),
                    Expect::Float(vec![c.linear_value_heads, h]),
                );
                add(
                    format!("{l}in_proj_b.weight"),
                    Expect::Float(vec![c.linear_value_heads, h]),
                );
                add(
                    format!("{l}conv1d.weight"),
                    Expect::Float(vec![2 * key_dim + value_dim, 1, c.conv_kernel]),
                );
                add(
                    format!("{l}A_log"),
                    Expect::Float(vec![c.linear_value_heads]),
                );
                add(
                    format!("{l}dt_bias"),
                    Expect::Float(vec![c.linear_value_heads]),
                );
                add(
                    format!("{l}norm.weight"),
                    Expect::Float(vec![c.linear_value_dim]),
                );
            }
            Qwen4ExpLayerKind::Full => {
                let a = format!("{p}self_attn.");
                add(format!("{a}q_proj"), affine(2 * c.heads * c.head_dim, h));
                add(format!("{a}k_proj"), affine(c.kv_heads * c.head_dim, h));
                add(format!("{a}v_proj"), affine(c.kv_heads * c.head_dim, h));
                add(format!("{a}o_proj"), affine(h, c.heads * c.head_dim));
                add(format!("{a}q_norm.weight"), Expect::Float(vec![c.head_dim]));
                add(format!("{a}k_norm.weight"), Expect::Float(vec![c.head_dim]));
                add(
                    format!("{a}indexer.index_qk_proj"),
                    affine((c.indexer_heads + 1) * c.indexer_head_dim, h),
                );
                add(
                    format!("{a}indexer.q_layernorm.weight"),
                    Expect::Float(vec![c.indexer_head_dim]),
                );
                add(
                    format!("{a}indexer.k_layernorm.weight"),
                    Expect::Float(vec![c.indexer_head_dim]),
                );
            }
        }
        if layer == c.ple_layer {
            let e = format!("{p}ple.");
            add(format!("{e}key_proj"), affine(hc, c.ple_dim));
            add(format!("{e}value_proj"), affine(h, c.ple_dim));
            for norm in ["norm_conv", "norm_key", "norm_query"] {
                add(format!("{e}{norm}.weight"), Expect::Float(vec![hc]));
            }
            add(
                format!("{e}conv1d.weight"),
                Expect::Float(vec![hc, 1, c.ple_conv_kernel]),
            );
            let heads = c.ngram_heads();
            add(
                format!("{e}ple_embedding.layer_multipliers"),
                Expect::Int(vec![c.ngram_size]),
            );
            add(
                format!("{e}ple_embedding.ngram_heads_vocab_sizes"),
                Expect::Int(vec![heads]),
            );
            add(
                format!("{e}ple_embedding.ngram_heads_offsets"),
                Expect::Int(vec![heads]),
            );
            add(
                format!("{e}ple_embedding.ngram_embedding.weight_scale"),
                Expect::Float(vec![1]),
            );
        }
    }
    out
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
            .ok_or_else(|| cfg_err("quantization.group_size"))? as usize;
        let default_bits = quant
            .get("bits")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| cfg_err("quantization.bits"))? as usize;
        let mut overrides = HashMap::new();
        for (key, value) in quant {
            if let Some(bits) = value.get("bits").and_then(|v| v.as_u64()) {
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
            for (tensor, entry) in &header.tensors {
                if weight_map.get(tensor).and_then(|v| v.as_str()) != Some(name.as_str()) {
                    return Err(cfg_err(format!("{tensor}: index et en-tête divergent")));
                }
                let size = dtype_size(entry.dtype).ok_or_else(|| {
                    cfg_err(format!("{tensor}: dtype {:?} inattendu", entry.dtype))
                })?;
                let expected = entry
                    .shape
                    .iter()
                    .try_fold(size, |acc, d| acc.checked_mul(*d))
                    .ok_or_else(|| cfg_err(format!("{tensor}: forme trop grande")))?;
                let [start, end] = entry.data_offsets;
                let physical = header.data_start.checked_add(end as u64);
                if end.checked_sub(start) != Some(expected) || physical.is_none_or(|p| p > len) {
                    return Err(cfg_err(format!("{tensor}: plage incohérente")));
                }
                entries.insert(tensor.clone(), (shard_index, entry.clone()));
            }
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
        let width = shape.1 as usize;
        if width * config.ngram_heads() != config.ple_dim || !scale.is_finite() || scale <= 0.0 {
            return Err(cfg_err(format!(
                "table n-gram largeur {width}, échelle {scale} incohérentes"
            )));
        }
        let path = dir.join("ngram").join(file_name);
        let file = File::open(&path).map_err(|e| io(&path, e))?;
        let len = file.metadata().map_err(|e| io(&path, e))?.len();
        if len != shape.0 * shape.1 {
            return Err(cfg_err(format!(
                "table n-gram: {len} octets, {} attendus",
                shape.0 * shape.1
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
        let mut seen = BTreeSet::new();
        for (name, expect) in expected_tensors(&self.config) {
            match expect {
                Expect::Affine { rows, cols, stack } => {
                    let bits = self.bits_of(&name);
                    let groups = cols / self.group_size;
                    if cols % self.group_size != 0 || (cols * bits) % 32 != 0 {
                        return Err(cfg_err(format!("{name}: {cols} colonnes / u{bits}")));
                    }
                    let lead: Vec<usize> = stack.into_iter().collect();
                    let shape = |last| {
                        let mut s = lead.clone();
                        s.extend([rows, last]);
                        s
                    };
                    self.expect_entry(
                        &format!("{name}.weight"),
                        Dtype::U32,
                        &shape(cols * bits / 32),
                    )?;
                    self.expect_entry(&format!("{name}.scales"), Dtype::BF16, &shape(groups))?;
                    self.expect_entry(&format!("{name}.biases"), Dtype::BF16, &shape(groups))?;
                    for suffix in [".weight", ".scales", ".biases"] {
                        seen.insert(format!("{name}{suffix}"));
                    }
                }
                Expect::Float(shape) => {
                    let (_, entry) = self.entry(&name)?;
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
                    self.expect_entry(&name, Dtype::I64, &shape)?;
                    seen.insert(name);
                }
            }
        }
        if let Some(extra) = self.index.keys().find(|name| !seen.contains(*name)) {
            return Err(cfg_err(format!("tenseur inattendu: {extra}")));
        }
        Ok(())
    }

    fn expect_entry(&self, name: &str, dtype: Dtype, shape: &[usize]) -> Result<()> {
        let (_, entry) = self.entry(name)?;
        if entry.dtype != dtype || entry.shape != shape {
            return Err(cfg_err(format!(
                "{name}: {:?} {:?}, attendu {dtype:?} {shape:?}",
                entry.dtype, entry.shape
            )));
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

    fn validate_mtp_metadata(&self) -> Result<()> {
        let path = self.dir.join("mtp.safetensors");
        let header = read_shard_header(&path)?;
        let c = &self.config;
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            let name = format!("mtp.layers.0.mlp.switch_mlp.{proj}.weight");
            let entry = header
                .tensors
                .get(&name)
                .ok_or_else(|| cfg_err(format!("MTP: {name} absent")))?;
            if entry.shape.first() != Some(&c.experts) {
                return Err(cfg_err(format!("MTP: {name} forme {:?}", entry.shape)));
            }
        }
        for name in [
            "mtp.fc_embedding.weight",
            "mtp.fc_hidden.weight",
            "mtp.pre_fc_norm_hidden.weight",
        ] {
            if !header.tensors.contains_key(name) {
                return Err(cfg_err(format!("MTP: {name} absent")));
            }
        }
        Ok(())
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
