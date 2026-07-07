//! Chargement des poids Qwen depuis safetensors et configuration.

use crate::{
    decoder::DecoderTensor,
    quantization::bytes_to_u32,
    safetensor::{bytes_to_dense_f32, tensor_from_safetensor_parts},
    AffineQuantizedTensor, CausalDecoder, InferError, LinearWeight, ModelAssets, ModelConfig,
    QuantConfig, Result, Tensor, WeightCatalog,
};
use safetensors::Dtype;
use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const REQUIRED_DECODER_WEIGHTS: &[&str] = &["embed_tokens.weight", "norm.weight"];

/// Clé cible de la tête LM, partagée entre chargement et vérification de contrat.
const LM_HEAD_WEIGHT: &str = "lm_head.weight";

/// Clé cible des embeddings d'entrée, source du tying de la tête LM.
const EMBED_TOKENS_WEIGHT: &str = "embed_tokens.weight";

const FULL_ATTENTION_LAYER_WEIGHTS: &[&str] = &[
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
];

const LINEAR_ATTENTION_LAYER_WEIGHTS: &[&str] = &[
    "input_layernorm.weight",
    "linear_attn.in_proj_qkv.weight",
    "linear_attn.in_proj_z.weight",
    "linear_attn.in_proj_b.weight",
    "linear_attn.in_proj_a.weight",
    "linear_attn.out_proj.weight",
    "linear_attn.conv1d.weight",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm.weight",
];

const OPTIONAL_LAYER_WEIGHTS: &[&str] = &[
    "self_attn.q_proj.bias",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.bias",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.bias",
    "mlp.up_proj.bias",
    "mlp.down_proj.bias",
    "lm_head.bias",
];

const MLP_LAYER_WEIGHTS: &[&str] = &[
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
];

// Double norme feed-forward propre à Gemma (encadre le MLP, en plus de la
// post_attention_layernorm partagée).
const GEMMA_FFN_NORM_WEIGHTS: &[&str] = &[
    "pre_feedforward_layernorm.weight",
    "post_feedforward_layernorm.weight",
];

const GEMMA4_PARALLEL_MOE_WEIGHTS: &[&str] = &[
    "router.proj.weight",
    "router.scale",
    "router.per_expert_scale",
    "experts.switch_glu.gate_proj.weight",
    "experts.switch_glu.up_proj.weight",
    "experts.switch_glu.down_proj.weight",
    "pre_feedforward_layernorm_2.weight",
    "post_feedforward_layernorm_1.weight",
    "post_feedforward_layernorm_2.weight",
    "layer_scalar",
];

const MOE_LAYER_WEIGHTS: &[&str] = &[
    "post_attention_layernorm.weight",
    "mlp.gate.weight",
    "mlp.switch_mlp.gate_proj.weight",
    "mlp.switch_mlp.up_proj.weight",
    "mlp.switch_mlp.down_proj.weight",
];

const SHARED_EXPERT_LAYER_WEIGHTS: &[&str] = &[
    "mlp.shared_expert_gate.weight",
    "mlp.shared_expert.gate_proj.weight",
    "mlp.shared_expert.up_proj.weight",
    "mlp.shared_expert.down_proj.weight",
];

const FP8_SCALE_BLOCK: usize = 128;

mod quantized;
mod shape_validation;

#[cfg(test)]
use self::quantized::apply_fp8_scales;
use self::quantized::{
    is_fp8_weight, quantized_contract_shape, tensor_from_entry, validate_dense_contract_dtype,
    validate_optional_fp8_scale_inv,
};
use self::shape_validation::{validate_decoder_meta_shapes, validate_decoder_shapes};

/// Alias rétro-compatible de [`DecoderContract`].
pub type QwenDecoderContract = DecoderContract;

/// Résumé de validation header-only d'un décodeur causal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecoderContract {
    /// Nombre de shards safetensors inspectés.
    pub shard_count: usize,
    /// Nombre de tenseurs catalogués dans les headers safetensors.
    pub catalog_tensors: usize,
    /// Nombre de specs obligatoires attendues par la config.
    pub required_specs: usize,
    /// Nombre de specs optionnelles connues par le chargeur.
    pub optional_specs: usize,
    /// Nombre de specs présentes dans les headers.
    pub present_specs: usize,
    /// La config annonce au moins une couche MTP.
    pub mtp_declared: bool,
    /// Un sidecar MTP exploitable est présent (`mtp.fc.weight` + tenseurs `mtp.*`).
    pub mtp_weights_present: bool,
    /// Nombre de tenseurs MTP détectés dans le sidecar.
    pub mtp_tensor_count: usize,
}

/// Charge un décodeur causal depuis les assets locaux.
///
/// Générique pour la famille SwiGLU + RMSNorm + RoPE + GQA (Qwen, Llama,
/// Mistral, Mixtral) : les features (MoE, attention linéaire hybride, q/k-norm,
/// biais, tying des embeddings) sont détectées depuis la config et la présence
/// des poids, sans hypothèse `model_type`.
///
/// # Errors
///
/// Renvoie une erreur si le modèle sort du périmètre minimal supporté.
pub fn load_causal_decoder(assets: &ModelAssets) -> Result<CausalDecoder> {
    load_causal_decoder_from_shards(&assets.config, &assets.shards, &assets.catalog)
}

/// Alias rétro-compatible de [`load_causal_decoder`] (ancien nom Qwen-spécifique).
///
/// # Errors
///
/// Renvoie une erreur si le modèle sort du périmètre minimal supporté.
pub fn load_qwen_causal_decoder(assets: &ModelAssets) -> Result<CausalDecoder> {
    load_causal_decoder(assets)
}

/// Vérifie le contrat du décodeur sans charger les payloads de poids.
///
/// # Errors
///
/// Renvoie une erreur si un poids requis, une forme ou un dtype est incompatible.
pub fn verify_decoder_contract(assets: &ModelAssets) -> Result<DecoderContract> {
    let mut contract =
        verify_decoder_contract_from_shards(&assets.config, &assets.shards, &assets.catalog)?;
    contract.mtp_declared = assets.config.mtp_num_hidden_layers.unwrap_or(0) > 0;
    contract.mtp_weights_present = assets.mtp.is_available();
    contract.mtp_tensor_count = assets.mtp.tensor_count;
    Ok(contract)
}

/// Alias rétro-compatible de [`verify_decoder_contract`].
///
/// # Errors
///
/// Renvoie une erreur si un poids requis, une forme ou un dtype est incompatible.
pub fn verify_qwen_decoder_contract(assets: &ModelAssets) -> Result<DecoderContract> {
    verify_decoder_contract(assets)
}

/// Vérifie un contrat de décodeur depuis des shards safetensors.
///
/// # Errors
///
/// Renvoie une erreur si les headers ne satisfont pas le contrat du décodeur.
pub fn verify_decoder_contract_from_shards(
    config: &ModelConfig,
    shards: &[PathBuf],
    catalog: &WeightCatalog,
) -> Result<DecoderContract> {
    validate_supported_config(config, catalog)?;
    let prefixes = QwenPrefixes::detect(catalog);
    let specs = decoder_specs(config, &prefixes, catalog);
    let headers = read_shard_headers(shards)?;
    let metas = decoder_contract_metas(config, &specs, &headers, catalog)?;
    validate_decoder_meta_shapes(config, &metas)?;
    let required_specs = specs.iter().filter(|spec| spec.required).count();
    let present_specs = metas.len();
    Ok(DecoderContract {
        shard_count: shards.len(),
        catalog_tensors: catalog.tensor_count(),
        required_specs,
        optional_specs: specs.len().saturating_sub(required_specs),
        present_specs,
        mtp_declared: config.mtp_num_hidden_layers.unwrap_or(0) > 0,
        mtp_weights_present: false,
        mtp_tensor_count: 0,
    })
}

/// Alias rétro-compatible de [`verify_decoder_contract_from_shards`].
///
/// # Errors
///
/// Renvoie une erreur si les headers ne satisfont pas le contrat du décodeur.
pub fn verify_qwen_decoder_contract_from_shards(
    config: &ModelConfig,
    shards: &[PathBuf],
    catalog: &WeightCatalog,
) -> Result<DecoderContract> {
    verify_decoder_contract_from_shards(config, shards, catalog)
}

/// Charge un décodeur causal depuis des shards safetensors.
///
/// # Errors
///
/// Renvoie une erreur si les poids attendus manquent ou ne sont pas F32/BF16/F16.
pub fn load_causal_decoder_from_shards(
    config: &ModelConfig,
    shards: &[PathBuf],
    catalog: &WeightCatalog,
) -> Result<CausalDecoder> {
    validate_supported_config(config, catalog)?;
    let prefixes = QwenPrefixes::detect(catalog);
    let tensors = load_decoder_tensors(config, shards, catalog, &prefixes)?;
    validate_decoder_shapes(config, &tensors)?;
    CausalDecoder::from_decoder_tensors(tensors, config.into())
}

/// Alias rétro-compatible de [`load_causal_decoder_from_shards`].
///
/// # Errors
///
/// Renvoie une erreur si les poids attendus manquent ou ne sont pas F32/BF16/F16.
pub fn load_qwen_causal_decoder_from_shards(
    config: &ModelConfig,
    shards: &[PathBuf],
    catalog: &WeightCatalog,
) -> Result<CausalDecoder> {
    load_causal_decoder_from_shards(config, shards, catalog)
}

fn validate_supported_config(config: &ModelConfig, catalog: &WeightCatalog) -> Result<()> {
    // Gemma 2 déclare un softcapping (tanh) des scores d'attention et des logits
    // finaux que le forward n'implémente pas : refuser net plutôt que générer du
    // charabia. Gemma 3 sérialise ces clés à `null` (qk-norm a remplacé le cap).
    if config.attn_logit_softcapping.is_some_and(|cap| cap != 0.0)
        || (!config.is_gemma4() && config.final_logit_softcapping.is_some_and(|cap| cap != 0.0))
    {
        return Err(InferError::Config(format!(
            "logit softcapping non supporté (Gemma 2): attn={:?}, final={:?}",
            config.attn_logit_softcapping, config.final_logit_softcapping
        )));
    }
    validate_rope_scaling(config)?;
    if let Some(quant) = &config.quantization {
        validate_affine_quantization(quant)?;
    }
    if config.is_hybrid() {
        validate_hybrid_config(config)?;
    }
    if let Some(layer) = first_layer_outside_config(catalog.keys(), config.num_hidden_layers) {
        return Err(InferError::Config(format!(
            "poids de couche hors config: layer {layer}, num_hidden_layers={}",
            config.num_hidden_layers
        )));
    }
    Ok(())
}

/// Valide la section `rope_scaling` déclarée par la config.
///
/// Seul le type `linear` est implémenté (Gemma 3 ≥4B : ×8 des positions des
/// couches globales). Pour Gemma, un autre type produirait du charabia → refus
/// net, comme le softcapping Gemma 2. Hors Gemma, le statu quo du chargeur
/// générique est préservé : les types non implémentés (llama3 de Llama 3.2,
/// yarn…) restent ignorés.
fn validate_rope_scaling(config: &ModelConfig) -> Result<()> {
    let Some(scaling) = config.rope_scaling.as_ref() else {
        return Ok(());
    };
    let scaling_type = scaling.scaling_type();
    if scaling_type == "linear" && config.rope_position_scale().is_none() {
        return Err(InferError::Config(format!(
            "rope_scaling linear sans facteur exploitable: {:?}",
            scaling.factor
        )));
    }
    if config.is_gemma() && !matches!(scaling_type, "linear" | "default") {
        return Err(InferError::Config(format!(
            "rope_scaling type {scaling_type} non supporté pour Gemma"
        )));
    }
    Ok(())
}

fn validate_hybrid_config(config: &ModelConfig) -> Result<()> {
    let interval = config
        .full_attention_interval
        .ok_or_else(|| InferError::Config("hybride sans full_attention_interval".to_string()))?;
    if interval == 0 {
        return Err(InferError::Config(
            "full_attention_interval nul".to_string(),
        ));
    }
    for (name, value) in [
        ("linear_num_key_heads", config.linear_num_key_heads),
        ("linear_num_value_heads", config.linear_num_value_heads),
        ("linear_key_head_dim", config.linear_key_head_dim),
        ("linear_value_head_dim", config.linear_value_head_dim),
        ("linear_conv_kernel_dim", config.linear_conv_kernel_dim),
    ] {
        if value.unwrap_or(0) == 0 {
            return Err(InferError::Config(format!(
                "config hybride sans {name} exploitable"
            )));
        }
    }
    Ok(())
}

fn validate_affine_quantization(quant: &QuantConfig) -> Result<()> {
    if quant
        .quant_method
        .as_deref()
        .is_some_and(|method| method.eq_ignore_ascii_case("fp8"))
    {
        if !matches!(quant.fmt.as_deref(), None | Some("e4m3") | Some("E4M3")) {
            return Err(InferError::Config(format!(
                "format FP8 non supporté: {:?}",
                quant.fmt
            )));
        }
        return Ok(());
    }
    let (group_size, bits) = quant_params(quant)?;
    if !matches!(bits, 4 | 8) {
        return Err(InferError::Config(format!(
            "quantification affine bits={bits} non supportée"
        )));
    }
    if group_size == 0 {
        return Err(InferError::Config(
            "quantification affine avec group_size nul".to_string(),
        ));
    }
    Ok(())
}

fn quant_params(quant: &QuantConfig) -> Result<(usize, usize)> {
    quant_params_for(quant, "")
}

fn quant_params_for(quant: &QuantConfig, source: &str) -> Result<(usize, usize)> {
    let base = source.strip_suffix(".weight").unwrap_or(source);
    let group_size = quant_override_usize(quant, base, "group_size")?
        .or(quant.group_size)
        .ok_or_else(|| InferError::Config("quantification affine sans group_size".to_string()))?;
    let bits = quant_override_usize(quant, base, "bits")?
        .or(quant.bits)
        .ok_or_else(|| InferError::Config("quantification affine sans bits".to_string()))?;
    Ok((group_size, bits))
}

fn quant_override_usize(quant: &QuantConfig, base: &str, field: &str) -> Result<Option<usize>> {
    let Some(value) = quant.extra.get(base).and_then(|entry| entry.get(field)) else {
        return Ok(None);
    };
    let Some(raw) = value.as_u64() else {
        return Err(InferError::Config(format!(
            "quantification {base}.{field} non numérique"
        )));
    };
    usize::try_from(raw).map(Some).map_err(|_| {
        InferError::Config(format!(
            "quantification {base}.{field} hors plage usize: {raw}"
        ))
    })
}

fn first_layer_outside_config(keys: &[String], num_hidden_layers: usize) -> Option<usize> {
    keys.iter()
        .filter_map(|key| layer_index(key))
        .find(|layer| *layer >= num_hidden_layers)
}

fn layer_index(key: &str) -> Option<usize> {
    let marker = ".layers.";
    let start = key.find(marker)? + marker.len();
    let tail = key.get(start..)?;
    let end = tail.find('.')?;
    tail.get(..end)?.parse::<usize>().ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct QwenPrefixes {
    weight: String,
    lm_head: String,
}

impl QwenPrefixes {
    fn detect(catalog: &WeightCatalog) -> Self {
        if !catalog.weight_prefix().is_empty() || !catalog.lm_head_prefix().is_empty() {
            return Self {
                weight: catalog.weight_prefix().to_string(),
                lm_head: catalog.lm_head_prefix().to_string(),
            };
        }

        for (weight, lm_head) in [
            ("language_model.model.", "language_model.lm_head."),
            ("model.language_model.", "lm_head."),
            ("model.", "lm_head."),
            ("", ""),
        ] {
            if catalog.contains(&format!("{weight}embed_tokens.weight"))
                || catalog.contains(&format!("{weight}layers.0.input_layernorm.weight"))
            {
                return Self {
                    weight: weight.to_string(),
                    lm_head: lm_head.to_string(),
                };
            }
        }

        Self {
            weight: String::new(),
            lm_head: String::new(),
        }
    }

    fn source_for(&self, target: &str) -> String {
        if target.starts_with("lm_head.") {
            if self.lm_head.is_empty() {
                target.to_string()
            } else {
                format!(
                    "{}{}",
                    self.lm_head,
                    target.strip_prefix("lm_head.").unwrap_or(target)
                )
            }
        } else {
            format!("{}{}", self.weight, target)
        }
    }
}

#[derive(Debug, Clone)]
struct TensorSpec {
    source: String,
    target: String,
    required: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TensorMeta {
    shape: Vec<usize>,
}

#[derive(Debug)]
struct ShardHeader {
    path: PathBuf,
    data_start: u64,
    tensors: HashMap<String, ShardTensorEntry>,
}

#[derive(Debug, Clone)]
struct TensorEntryRef {
    shard_index: usize,
    entry: ShardTensorEntry,
}

#[derive(Debug, Clone, serde::Deserialize)]
struct ShardTensorEntry {
    dtype: Dtype,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn load_decoder_tensors(
    config: &ModelConfig,
    shards: &[PathBuf],
    catalog: &WeightCatalog,
    prefixes: &QwenPrefixes,
) -> Result<HashMap<String, DecoderTensor>> {
    let specs = decoder_specs(config, prefixes, catalog);
    let headers = read_shard_headers(shards)?;
    let entries = index_shard_entries(&headers)?;
    let mut out = HashMap::new();

    for spec in &specs {
        let Some(entry_ref) = entries.get(&spec.source) else {
            if spec.required {
                let missing = if catalog.contains(&spec.source) {
                    spec.source.clone()
                } else {
                    spec.target.clone()
                };
                return Err(InferError::MissingWeight(missing));
            }
            continue;
        };
        if out.contains_key(&spec.target) {
            return Err(InferError::Config(format!(
                "poids dupliqué pour {} depuis {}",
                spec.target, spec.source
            )));
        }
        let tensor = tensor_from_entry(config, spec, entry_ref, &headers, &entries)?;
        out.insert(spec.target.clone(), tensor);
    }

    tie_lm_head_to_embeddings(config, &mut out);
    bake_gemma_norm_offset(config, &mut out);
    Ok(out)
}

/// Intègre l'offset `1 + weight` des RMSNorm Gemma 2/3 dans les poids.
///
/// Gemma 4 sérialise des poids RMSNorm directs (`x_normé · weight`), donc il
/// emprunte le même chemin que Qwen/Llama pour éviter de doubler les échelles.
fn bake_gemma_norm_offset(config: &ModelConfig, tensors: &mut HashMap<String, DecoderTensor>) {
    if !config.is_gemma() || config.is_gemma4() {
        return;
    }
    for (key, tensor) in tensors.iter_mut() {
        if !key.ends_with("norm.weight") {
            continue;
        }
        if let DecoderTensor::Dense(weight) = tensor {
            *weight = weight.map(|value| value + 1.0);
        }
    }
}

/// Substitue les embeddings d'entrée à la tête LM pour les modèles à
/// embeddings liés (`tie_word_embeddings`) qui ne stockent pas de `lm_head.weight`.
///
/// Llama-3.2-1B/3B, Qwen3-0.6B, Gemma… réutilisent `embed_tokens.weight` comme
/// projection finale ; mlx_lm ne sérialise donc pas de tête séparée. Le clone est
/// borné aux modèles tied : les Qwen prod (27B/30B/35B) stockent leur `lm_head`
/// et empruntent ce chemin sans aucune copie (byte-identité préservée).
fn tie_lm_head_to_embeddings<T: Clone>(config: &ModelConfig, tensors: &mut HashMap<String, T>) {
    if !config.tie_word_embeddings || tensors.contains_key(LM_HEAD_WEIGHT) {
        return;
    }
    if let Some(embed) = tensors.get(EMBED_TOKENS_WEIGHT).cloned() {
        tensors.insert(LM_HEAD_WEIGHT.to_string(), embed);
    }
}

fn read_shard_headers(shards: &[PathBuf]) -> Result<Vec<ShardHeader>> {
    shards
        .iter()
        .map(|shard| read_shard_header(shard))
        .collect()
}

fn index_shard_entries(headers: &[ShardHeader]) -> Result<HashMap<String, TensorEntryRef>> {
    let mut entries = HashMap::new();
    for (shard_index, header) in headers.iter().enumerate() {
        for (name, entry) in &header.tensors {
            if entries
                .insert(
                    name.clone(),
                    TensorEntryRef {
                        shard_index,
                        entry: entry.clone(),
                    },
                )
                .is_some()
            {
                return Err(InferError::Config(format!(
                    "poids dupliqué dans les headers: {name}"
                )));
            }
        }
    }
    Ok(entries)
}

fn read_shard_header(path: &Path) -> Result<ShardHeader> {
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
    let raw: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header).map_err(|source| InferError::Json {
            path: path.to_path_buf(),
            source,
        })?;
    let mut tensors = HashMap::with_capacity(raw.len());
    for (name, value) in raw {
        if name == "__metadata__" {
            continue;
        }
        let entry: ShardTensorEntry =
            serde_json::from_value(value).map_err(|source| InferError::Json {
                path: path.to_path_buf(),
                source,
            })?;
        if entry.data_offsets[1] < entry.data_offsets[0] {
            return Err(InferError::SafetensorsHeader {
                path: path.to_path_buf(),
                message: format!("offsets invalides pour {name}: {:?}", entry.data_offsets),
            });
        }
        tensors.insert(name, entry);
    }
    Ok(ShardHeader {
        path: path.to_path_buf(),
        data_start: 8 + header_len,
        tensors,
    })
}

fn decoder_contract_metas(
    config: &ModelConfig,
    specs: &[TensorSpec],
    headers: &[ShardHeader],
    catalog: &WeightCatalog,
) -> Result<HashMap<String, TensorMeta>> {
    let entries = index_shard_entries(headers)?;
    let mut out = HashMap::new();
    for spec in specs {
        let Some(entry_ref) = entries.get(&spec.source) else {
            if spec.required {
                let missing = if catalog.contains(&spec.source) {
                    spec.source.clone()
                } else {
                    spec.target.clone()
                };
                return Err(InferError::MissingWeight(missing));
            }
            continue;
        };
        let shape = logical_contract_shape(config, spec, entry_ref, &entries)?;
        out.insert(spec.target.clone(), TensorMeta { shape });
    }
    tie_lm_head_to_embeddings(config, &mut out);
    Ok(out)
}

fn logical_contract_shape(
    config: &ModelConfig,
    spec: &TensorSpec,
    entry_ref: &TensorEntryRef,
    entries: &HashMap<String, TensorEntryRef>,
) -> Result<Vec<usize>> {
    let entry = &entry_ref.entry;
    if entry.dtype == Dtype::U32 && spec.source.ends_with(".weight") {
        return quantized_contract_shape(config, spec, entry, entries);
    }
    validate_dense_contract_dtype(&spec.source, entry.dtype)?;
    if is_fp8_weight(entry.dtype, &spec.source) {
        validate_optional_fp8_scale_inv(spec, entry, entries)?;
    }
    Ok(entry.shape.clone())
}

fn decoder_specs(
    config: &ModelConfig,
    prefixes: &QwenPrefixes,
    catalog: &WeightCatalog,
) -> Vec<TensorSpec> {
    let mut specs = specs_for(
        prefixes,
        REQUIRED_DECODER_WEIGHTS
            .iter()
            .map(|target| (*target).to_string()),
        true,
    )
    .collect::<Vec<_>>();
    // La tête LM n'est requise que pour les modèles non liés ; les modèles
    // `tie_word_embeddings` réutilisent `embed_tokens.weight` (substitué après chargement).
    specs.extend(specs_for(
        prefixes,
        [LM_HEAD_WEIGHT.to_string()],
        !config.tie_word_embeddings,
    ));
    for layer in 0..config.num_hidden_layers {
        let required_weights = if config.is_full_attention_layer(layer) {
            FULL_ATTENTION_LAYER_WEIGHTS
        } else {
            LINEAR_ATTENTION_LAYER_WEIGHTS
        };
        let layer_required = required_weights
            .iter()
            .filter(|suffix| {
                !(config.attention_k_eq_v
                    && config.is_gemma4_full_layer(layer)
                    && **suffix == "self_attn.v_proj.weight")
            })
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        specs.extend(specs_for(prefixes, layer_required, true));

        let layer_mlp = MLP_LAYER_WEIGHTS
            .iter()
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        let has_mlp = layer_mlp
            .iter()
            .filter(|target| !target.ends_with("post_attention_layernorm.weight"))
            .any(|target| catalog.contains(&prefixes.source_for(target)));
        if has_mlp {
            specs.extend(specs_for(prefixes, layer_mlp, true));
        }

        let layer_gemma4_moe = GEMMA4_PARALLEL_MOE_WEIGHTS
            .iter()
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        let has_gemma4_moe = config.enable_moe_block
            || layer_gemma4_moe
                .iter()
                .any(|target| catalog.contains(&prefixes.source_for(target)));
        if has_gemma4_moe {
            specs.extend(specs_for(prefixes, layer_gemma4_moe, true));
        }

        let layer_moe = MOE_LAYER_WEIGHTS
            .iter()
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        let has_moe = !config.enable_moe_block
            && (config.is_moe()
                || layer_moe
                    .iter()
                    .filter(|target| !target.ends_with("post_attention_layernorm.weight"))
                    .any(|target| catalog.contains(&prefixes.source_for(target))));
        if has_moe {
            specs.extend(specs_for(prefixes, layer_moe, true));
        }
        let layer_shared_expert = SHARED_EXPERT_LAYER_WEIGHTS
            .iter()
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        let has_shared_expert = config.shared_expert_intermediate_size.is_some()
            || layer_shared_expert
                .iter()
                .any(|target| catalog.contains(&prefixes.source_for(target)));
        if has_shared_expert {
            specs.extend(specs_for(prefixes, layer_shared_expert, true));
        }

        // Gemma : double norme feed-forward, requise quand l'architecture est Gemma.
        let gemma_ffn_norms = GEMMA_FFN_NORM_WEIGHTS
            .iter()
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        specs.extend(specs_for(prefixes, gemma_ffn_norms, config.is_gemma()));

        let layer_optional = OPTIONAL_LAYER_WEIGHTS
            .iter()
            .filter(|suffix| {
                config.is_full_attention_layer(layer) || !suffix.starts_with("self_attn.")
            })
            .filter(|suffix| !suffix.starts_with("lm_head."))
            .map(|suffix| layer_target(layer, suffix))
            .collect::<Vec<_>>();
        specs.extend(specs_for(prefixes, layer_optional, false));
    }
    specs.extend(specs_for(
        prefixes,
        ["lm_head.bias"].iter().map(|target| (*target).to_string()),
        false,
    ));
    specs
}

fn specs_for<'a>(
    prefixes: &'a QwenPrefixes,
    targets: impl IntoIterator<Item = String> + 'a,
    required: bool,
) -> impl Iterator<Item = TensorSpec> + 'a {
    targets.into_iter().map(move |target| TensorSpec {
        source: prefixes.source_for(&target),
        target,
        required,
    })
}

fn layer_target(layer: usize, suffix: &str) -> String {
    format!("layers.{layer}.{suffix}")
}

#[cfg(test)]
mod test_fixtures;
#[cfg(test)]
mod tests;
