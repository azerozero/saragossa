//! Validation et chargement des checkpoints draft DFlash.

use crate::{
    catalog::read_safetensors_keys, rms_norm, safetensor::bytes_to_dense_f32, ForwardRuntime,
    GatedMlp, InferError, Linear, ModelConfig, Result, Tensor,
};
use safetensors::{tensor::TensorView, SafeTensors};
use serde::Deserialize;
use std::f32;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq)]
pub struct DFlashDraftInfo {
    /// Repertoire racine du draft DFlash.
    pub draft_dir: PathBuf,
    /// Fichier de poids principal du draft.
    pub weight_path: PathBuf,
    /// Nombre de tenseurs presents dans le checkpoint draft.
    pub tensor_count: usize,
    /// Taille du bloc speculatif declaree par le draft.
    pub block_size: usize,
    /// Token masque utilise pour remplir la queue du bloc.
    pub mask_token_id: usize,
    /// Couches du trunk dont les hidden states alimentent le draft.
    pub target_layer_ids: Vec<usize>,
    /// Nombre de couches du mini-decodeur draft.
    pub num_hidden_layers: usize,
    /// Dimension cachee du draft.
    pub hidden_size: usize,
    /// Dimension cachee du MLP dense du draft.
    pub intermediate_size: usize,
    /// Nombre de tetes Q du draft.
    pub num_attention_heads: usize,
    /// Nombre de tetes KV du draft.
    pub num_key_value_heads: usize,
    /// Dimension d'une tete d'attention.
    pub head_dim: usize,
    /// Epsilon RMSNorm du draft.
    pub rms_norm_eps: f32,
    /// Base RoPE du draft.
    pub rope_theta: f32,
    /// Type d'attention de chaque couche draft.
    pub layer_types: Vec<String>,
    /// Fenetre des couches `sliding_attention`.
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone)]
pub struct DFlashDraft {
    pub info: DFlashDraftInfo,
    pub hidden_norm: Tensor,
    pub fc: Linear,
    pub layers: Vec<DFlashDraftLayer>,
    pub norm: Tensor,
}

#[derive(Debug, Clone)]
pub struct DFlashDraftLayer {
    pub input_norm: Tensor,
    pub attention: DFlashAttentionWeights,
    pub post_attention_norm: Tensor,
    pub mlp: GatedMlp,
}

#[derive(Debug, Clone)]
pub struct DFlashAttentionWeights {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
}

impl DFlashDraft {
    /// Projette les hidden states du trunk cible vers le contexte compact DFlash.
    ///
    /// L'entree attendue est une matrice `[seq, target_layers * hidden]`,
    /// c'est-a-dire les hidden states des couches cible concatenees par token.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la forme ne correspond pas au draft ou si la
    /// projection echoue sur le runtime fourni.
    pub fn project_target_hidden(
        &self,
        target_hidden: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let (_, cols) = target_hidden.as_matrix()?;
        let expected_cols = self
            .info
            .target_layer_ids
            .len()
            .checked_mul(self.info.hidden_size)
            .ok_or_else(|| InferError::Shape("draft DFlash target hidden deborde".to_string()))?;
        if cols != expected_cols {
            return Err(InferError::Dimension(format!(
                "draft DFlash target_hidden attendu [seq,{expected_cols}], recu {:?}",
                target_hidden.shape()
            )));
        }
        let projected = self.fc.forward_with_runtime(target_hidden, runtime)?;
        rms_norm(&projected, &self.hidden_norm, self.info.rms_norm_eps)
    }

    /// Execute le mini-decodeur DFlash sur un bloc de tokens bruites.
    ///
    /// L'entree `noise_embedding` est `[block, hidden]`, et `draft_context`
    /// est `[ctx, hidden]`, deja projete par [`Self::project_target_hidden`].
    /// Ce chemin CPU sans cache sert d'oracle fonctionnel avant le port Metal.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une forme diverge ou si un sous-bloc echoue.
    pub fn forward_projected_context(
        &self,
        noise_embedding: &Tensor,
        draft_context: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        expect_matrix_cols(
            noise_embedding,
            self.info.hidden_size,
            "draft noise_embedding",
        )?;
        expect_matrix_cols(draft_context, self.info.hidden_size, "draft_context")?;
        let mut hidden = noise_embedding.clone();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&self.info, layer_idx, &hidden, draft_context, runtime)?;
        }
        rms_norm(&hidden, &self.norm, self.info.rms_norm_eps)
    }
}

impl DFlashDraftLayer {
    fn forward(
        &self,
        info: &DFlashDraftInfo,
        layer_idx: usize,
        hidden_states: &Tensor,
        draft_context: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let normed = rms_norm(hidden_states, &self.input_norm, info.rms_norm_eps)?;
        let attn_out = self
            .attention
            .forward(info, layer_idx, &normed, draft_context, runtime)?;
        let attention_state = hidden_states.add(&attn_out)?;
        let mlp_input = rms_norm(
            &attention_state,
            &self.post_attention_norm,
            info.rms_norm_eps,
        )?;
        attention_state.add(&self.mlp.forward_with_runtime(&mlp_input, runtime)?)
    }
}

impl DFlashAttentionWeights {
    fn forward(
        &self,
        info: &DFlashDraftInfo,
        layer_idx: usize,
        hidden_states: &Tensor,
        draft_context: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Tensor> {
        let (block_len, hidden) = hidden_states.as_matrix()?;
        let (ctx_len, ctx_hidden) = draft_context.as_matrix()?;
        if hidden != info.hidden_size || ctx_hidden != info.hidden_size {
            return Err(InferError::Dimension(format!(
                "draft attention hidden incompatible hidden={hidden}, ctx={ctx_hidden}, attendu={}",
                info.hidden_size
            )));
        }
        let q = self.q_proj.forward_with_runtime(hidden_states, runtime)?;
        let q = rms_norm_heads(
            &q,
            info.num_attention_heads,
            info.head_dim,
            &self.q_norm,
            info.rms_norm_eps,
        )?;
        let q = apply_rope_heads(
            &q,
            info.num_attention_heads,
            info.head_dim,
            info.rope_theta,
            ctx_len,
        )?;

        let context_k = self.k_proj.forward_with_runtime(draft_context, runtime)?;
        let context_k = rms_norm_heads(
            &context_k,
            info.num_key_value_heads,
            info.head_dim,
            &self.k_norm,
            info.rms_norm_eps,
        )?;
        let context_k = apply_rope_heads(
            &context_k,
            info.num_key_value_heads,
            info.head_dim,
            info.rope_theta,
            0,
        )?;
        let context_v = self.v_proj.forward_with_runtime(draft_context, runtime)?;

        let noise_k = self.k_proj.forward_with_runtime(hidden_states, runtime)?;
        let noise_k = rms_norm_heads(
            &noise_k,
            info.num_key_value_heads,
            info.head_dim,
            &self.k_norm,
            info.rms_norm_eps,
        )?;
        let noise_k = apply_rope_heads(
            &noise_k,
            info.num_key_value_heads,
            info.head_dim,
            info.rope_theta,
            ctx_len,
        )?;
        let noise_v = self.v_proj.forward_with_runtime(hidden_states, runtime)?;

        let q_dim = info.num_attention_heads * info.head_dim;
        let mut out = vec![0.0_f32; block_len * q_dim];
        let sliding_window = info.layer_sliding_window(layer_idx);
        let scale = (info.head_dim as f32).sqrt().recip();
        for query_pos in 0..block_len {
            for head in 0..info.num_attention_heads {
                let kv_head = head / (info.num_attention_heads / info.num_key_value_heads);
                let q_base = query_pos * q_dim + head * info.head_dim;
                let mut scores = Vec::with_capacity(ctx_len + block_len);
                for key_pos in 0..ctx_len {
                    if !attention_position_allowed(ctx_len + query_pos, key_pos, sliding_window) {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let k_base = key_pos * info.num_key_value_heads * info.head_dim
                        + kv_head * info.head_dim;
                    scores.push(
                        dot(
                            &q.data()[q_base..q_base + info.head_dim],
                            &context_k.data()[k_base..k_base + info.head_dim],
                        ) * scale,
                    );
                }
                for key_pos in 0..block_len {
                    let absolute_key_pos = ctx_len + key_pos;
                    if key_pos > query_pos
                        || !attention_position_allowed(
                            ctx_len + query_pos,
                            absolute_key_pos,
                            sliding_window,
                        )
                    {
                        scores.push(f32::NEG_INFINITY);
                        continue;
                    }
                    let k_base = key_pos * info.num_key_value_heads * info.head_dim
                        + kv_head * info.head_dim;
                    scores.push(
                        dot(
                            &q.data()[q_base..q_base + info.head_dim],
                            &noise_k.data()[k_base..k_base + info.head_dim],
                        ) * scale,
                    );
                }
                let weights = softmax_scores(&scores)?;
                let out_base = query_pos * q_dim + head * info.head_dim;
                for key_pos in 0..ctx_len {
                    let weight = weights[key_pos];
                    if weight == 0.0 {
                        continue;
                    }
                    let v_base = key_pos * info.num_key_value_heads * info.head_dim
                        + kv_head * info.head_dim;
                    accumulate_scaled(
                        &mut out[out_base..out_base + info.head_dim],
                        &context_v.data()[v_base..v_base + info.head_dim],
                        weight,
                    );
                }
                for key_pos in 0..block_len {
                    let weight = weights[ctx_len + key_pos];
                    if weight == 0.0 {
                        continue;
                    }
                    let v_base = key_pos * info.num_key_value_heads * info.head_dim
                        + kv_head * info.head_dim;
                    accumulate_scaled(
                        &mut out[out_base..out_base + info.head_dim],
                        &noise_v.data()[v_base..v_base + info.head_dim],
                        weight,
                    );
                }
            }
        }
        let context = Tensor::from_vec(vec![block_len, q_dim], out)?;
        self.o_proj.forward_with_runtime(&context, runtime)
    }
}

impl DFlashDraftInfo {
    fn layer_sliding_window(&self, layer_idx: usize) -> Option<usize> {
        self.layer_types
            .get(layer_idx)
            .is_some_and(|kind| kind == "sliding_attention")
            .then_some(self.sliding_window)
            .flatten()
    }
}

#[derive(Debug, Deserialize)]
struct RawDFlashDraftConfig {
    #[serde(default)]
    architectures: Vec<String>,
    hidden_size: usize,
    num_hidden_layers: usize,
    num_target_layers: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    #[serde(default)]
    head_dim: Option<usize>,
    #[serde(default)]
    layer_types: Vec<String>,
    #[serde(default)]
    sliding_window: Option<usize>,
    rms_norm_eps: f32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: Option<RawRopeParameters>,
    dflash_config: RawDFlashConfig,
}

#[derive(Debug, Deserialize)]
struct RawDFlashConfig {
    block_size: usize,
    mask_token_id: usize,
    target_layer_ids: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct RawRopeParameters {
    rope_theta: Option<f32>,
}

/// Charge et valide un draft DFlash par rapport au trunk cible.
///
/// # Errors
///
/// Renvoie une erreur si le repertoire n'est pas un draft DFlash compatible
/// avec le nombre de couches et la dimension cachee du trunk.
pub fn load_dflash_draft_for_target(
    target: &ModelConfig,
    draft_dir: impl AsRef<Path>,
) -> Result<DFlashDraftInfo> {
    let draft_dir = draft_dir.as_ref();
    if !draft_dir.is_dir() {
        return Err(InferError::MissingArtifact {
            path: draft_dir.to_path_buf(),
            what: "dflash draft dir",
        });
    }
    let config_path = draft_dir.join("config.json");
    if !config_path.is_file() {
        return Err(InferError::MissingArtifact {
            path: config_path,
            what: "dflash config.json",
        });
    }
    let file = std::fs::File::open(&config_path).map_err(|source| InferError::Io {
        path: config_path.clone(),
        source,
    })?;
    let raw: RawDFlashDraftConfig =
        serde_json::from_reader(file).map_err(|source| InferError::Json {
            path: config_path.clone(),
            source,
        })?;

    if !raw
        .architectures
        .iter()
        .any(|name| name == "DFlashDraftModel")
    {
        return Err(InferError::Config(format!(
            "draft DFlash architecture absente: {:?}",
            raw.architectures
        )));
    }
    if raw.hidden_size != target.hidden_size {
        return Err(InferError::Dimension(format!(
            "draft DFlash hidden_size={} incompatible avec trunk hidden_size={}",
            raw.hidden_size, target.hidden_size
        )));
    }
    if raw.num_target_layers != target.num_hidden_layers {
        return Err(InferError::Dimension(format!(
            "draft DFlash num_target_layers={} incompatible avec trunk layers={}",
            raw.num_target_layers, target.num_hidden_layers
        )));
    }
    if raw.dflash_config.block_size == 0 {
        return Err(InferError::Config(
            "draft DFlash block_size doit etre > 0".to_string(),
        ));
    }
    if raw.num_hidden_layers == 0 {
        return Err(InferError::Config(
            "draft DFlash num_hidden_layers doit etre > 0".to_string(),
        ));
    }
    if raw.intermediate_size == 0 {
        return Err(InferError::Config(
            "draft DFlash intermediate_size doit etre > 0".to_string(),
        ));
    }
    if raw.num_attention_heads == 0 || raw.num_key_value_heads == 0 {
        return Err(InferError::Config(
            "draft DFlash attention heads doivent etre > 0".to_string(),
        ));
    }
    if raw.num_key_value_heads > raw.num_attention_heads {
        return Err(InferError::Config(format!(
            "draft DFlash num_key_value_heads={} > num_attention_heads={}",
            raw.num_key_value_heads, raw.num_attention_heads
        )));
    }
    if raw.num_attention_heads % raw.num_key_value_heads != 0 {
        return Err(InferError::Config(format!(
            "draft DFlash num_attention_heads={} non divisible par num_key_value_heads={}",
            raw.num_attention_heads, raw.num_key_value_heads
        )));
    }
    let head_dim = raw
        .head_dim
        .unwrap_or_else(|| raw.hidden_size / raw.num_attention_heads);
    if head_dim == 0 {
        return Err(InferError::Config(
            "draft DFlash head_dim doit etre > 0".to_string(),
        ));
    }
    if !raw.layer_types.is_empty() && raw.layer_types.len() != raw.num_hidden_layers {
        return Err(InferError::Config(format!(
            "draft DFlash layer_types={} incompatible avec num_hidden_layers={}",
            raw.layer_types.len(),
            raw.num_hidden_layers
        )));
    }
    if raw
        .layer_types
        .iter()
        .any(|kind| kind == "sliding_attention")
        && raw.sliding_window.unwrap_or(0) == 0
    {
        return Err(InferError::Config(
            "draft DFlash sliding_attention requiert sliding_window > 0".to_string(),
        ));
    }
    if !raw.rms_norm_eps.is_finite() || raw.rms_norm_eps <= 0.0 {
        return Err(InferError::Config(format!(
            "draft DFlash rms_norm_eps invalide: {}",
            raw.rms_norm_eps
        )));
    }
    let rope_theta = raw
        .rope_theta
        .or_else(|| {
            raw.rope_parameters
                .as_ref()
                .and_then(|rope| rope.rope_theta)
        })
        .ok_or_else(|| InferError::Config("draft DFlash rope_theta manquant".to_string()))?;
    if !rope_theta.is_finite() || rope_theta <= 0.0 {
        return Err(InferError::Config(format!(
            "draft DFlash rope_theta invalide: {rope_theta}"
        )));
    }
    if raw.dflash_config.target_layer_ids.is_empty() {
        return Err(InferError::Config(
            "draft DFlash target_layer_ids vide".to_string(),
        ));
    }
    for layer_id in &raw.dflash_config.target_layer_ids {
        if *layer_id >= target.num_hidden_layers {
            return Err(InferError::Dimension(format!(
                "draft DFlash target_layer_id {layer_id} hors trunk layers={}",
                target.num_hidden_layers
            )));
        }
    }

    let weight_path = draft_dir.join("model.safetensors");
    if !weight_path.is_file() {
        return Err(InferError::MissingArtifact {
            path: weight_path,
            what: "dflash model.safetensors",
        });
    }
    let keys = read_safetensors_keys(&weight_path)?;
    if !keys.iter().any(|key| key == "fc.weight") {
        return Err(InferError::MissingWeight("dflash fc.weight".to_string()));
    }

    Ok(DFlashDraftInfo {
        draft_dir: draft_dir.to_path_buf(),
        weight_path,
        tensor_count: keys.len(),
        block_size: raw.dflash_config.block_size,
        mask_token_id: raw.dflash_config.mask_token_id,
        target_layer_ids: raw.dflash_config.target_layer_ids,
        num_hidden_layers: raw.num_hidden_layers,
        hidden_size: raw.hidden_size,
        intermediate_size: raw.intermediate_size,
        num_attention_heads: raw.num_attention_heads,
        num_key_value_heads: raw.num_key_value_heads,
        head_dim,
        rms_norm_eps: raw.rms_norm_eps,
        rope_theta,
        layer_types: raw.layer_types,
        sliding_window: raw.sliding_window,
    })
}

/// Charge les poids d'un draft DFlash deja valide par rapport au trunk cible.
///
/// # Errors
///
/// Renvoie une erreur si un poids requis est absent, si un dtype n'est pas
/// convertible en f32 dense, ou si une forme diverge de la config draft.
pub fn load_dflash_draft_weights_for_target(
    target: &ModelConfig,
    draft_dir: impl AsRef<Path>,
) -> Result<DFlashDraft> {
    let info = load_dflash_draft_for_target(target, draft_dir)?;
    let bytes = std::fs::read(&info.weight_path).map_err(|source| InferError::Io {
        path: info.weight_path.clone(),
        source,
    })?;
    let st = SafeTensors::deserialize(&bytes).map_err(|source| InferError::Safetensors {
        path: info.weight_path.clone(),
        source,
    })?;
    let hidden = info.hidden_size;
    let fc_in = info
        .target_layer_ids
        .len()
        .checked_mul(hidden)
        .ok_or_else(|| InferError::Shape("draft DFlash fc input deborde".to_string()))?;
    let q_dim = info
        .num_attention_heads
        .checked_mul(info.head_dim)
        .ok_or_else(|| InferError::Shape("draft DFlash q_dim deborde".to_string()))?;
    let kv_dim = info
        .num_key_value_heads
        .checked_mul(info.head_dim)
        .ok_or_else(|| InferError::Shape("draft DFlash kv_dim deborde".to_string()))?;

    let hidden_norm = load_dflash_norm(&st, "hidden_norm.weight", hidden)?;
    let fc = load_dflash_linear(&st, "fc", hidden, fc_in)?;
    let mut layers = Vec::with_capacity(info.num_hidden_layers);
    for layer_idx in 0..info.num_hidden_layers {
        let prefix = format!("layers.{layer_idx}");
        let attention_prefix = format!("{prefix}.self_attn");
        let mlp_prefix = format!("{prefix}.mlp");
        let attention = DFlashAttentionWeights {
            q_proj: load_dflash_linear(&st, &format!("{attention_prefix}.q_proj"), q_dim, hidden)?,
            k_proj: load_dflash_linear(&st, &format!("{attention_prefix}.k_proj"), kv_dim, hidden)?,
            v_proj: load_dflash_linear(&st, &format!("{attention_prefix}.v_proj"), kv_dim, hidden)?,
            o_proj: load_dflash_linear(&st, &format!("{attention_prefix}.o_proj"), hidden, q_dim)?,
            q_norm: load_dflash_norm(
                &st,
                &format!("{attention_prefix}.q_norm.weight"),
                info.head_dim,
            )?,
            k_norm: load_dflash_norm(
                &st,
                &format!("{attention_prefix}.k_norm.weight"),
                info.head_dim,
            )?,
        };
        let mlp = GatedMlp::new(
            load_dflash_linear(
                &st,
                &format!("{mlp_prefix}.gate_proj"),
                info.intermediate_size,
                hidden,
            )?,
            load_dflash_linear(
                &st,
                &format!("{mlp_prefix}.up_proj"),
                info.intermediate_size,
                hidden,
            )?,
            load_dflash_linear(
                &st,
                &format!("{mlp_prefix}.down_proj"),
                hidden,
                info.intermediate_size,
            )?,
        );
        layers.push(DFlashDraftLayer {
            input_norm: load_dflash_norm(&st, &format!("{prefix}.input_layernorm.weight"), hidden)?,
            attention,
            post_attention_norm: load_dflash_norm(
                &st,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?,
            mlp,
        });
    }
    let norm = load_dflash_norm(&st, "norm.weight", hidden)?;
    Ok(DFlashDraft {
        info,
        hidden_norm,
        fc,
        layers,
        norm,
    })
}

fn load_dflash_linear(
    st: &SafeTensors,
    prefix: &str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<Linear> {
    let key = format!("{prefix}.weight");
    let tensor = load_dflash_tensor(st, &key)?;
    expect_shape(tensor.shape(), &[expected_rows, expected_cols], &key)?;
    Linear::new(tensor, None)
}

fn load_dflash_norm(st: &SafeTensors, key: &str, expected_dim: usize) -> Result<Tensor> {
    let tensor = load_dflash_tensor(st, key)?;
    expect_shape(tensor.shape(), &[expected_dim], key)?;
    Ok(tensor)
}

fn load_dflash_tensor(st: &SafeTensors, key: &str) -> Result<Tensor> {
    let view = st_tensor(st, key)?;
    Tensor::from_vec(
        view.shape().to_vec(),
        bytes_to_dense_f32(view.data(), view.dtype(), key)?,
    )
}

fn st_tensor<'a>(st: &'a SafeTensors, key: &str) -> Result<TensorView<'a>> {
    st.tensor(key)
        .map_err(|_| InferError::MissingWeight(format!("dflash {key}")))
}

fn expect_shape(actual: &[usize], expected: &[usize], key: &str) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(InferError::Dimension(format!(
            "draft DFlash {key} attendu {expected:?}, recu {actual:?}"
        )))
    }
}

fn expect_matrix_cols(tensor: &Tensor, expected_cols: usize, label: &str) -> Result<usize> {
    let (rows, cols) = tensor.as_matrix()?;
    if cols != expected_cols {
        return Err(InferError::Dimension(format!(
            "{label} attendu [seq,{expected_cols}], recu {:?}",
            tensor.shape()
        )));
    }
    Ok(rows)
}

fn rms_norm_heads(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = x.as_matrix()?;
    if heads == 0 || head_dim == 0 || cols != heads * head_dim {
        return Err(InferError::Dimension(format!(
            "draft RMSNorm heads invalide x={:?}, heads={heads}, head_dim={head_dim}",
            x.shape()
        )));
    }
    expect_shape(weight.shape(), &[head_dim], "draft qk_norm")?;
    let mut out = vec![0.0_f32; rows * cols];
    for row in 0..rows {
        for head in 0..heads {
            let base = row * cols + head * head_dim;
            let slice = &x.data()[base..base + head_dim];
            let mean_square =
                slice.iter().map(|value| value * value).sum::<f32>() / head_dim as f32;
            let inv = (mean_square + eps).sqrt().recip();
            for dim in 0..head_dim {
                out[base + dim] = slice[dim] * inv * weight.data()[dim];
            }
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

fn apply_rope_heads(
    x: &Tensor,
    heads: usize,
    head_dim: usize,
    theta: f32,
    position_offset: usize,
) -> Result<Tensor> {
    let (rows, cols) = x.as_matrix()?;
    if heads == 0 || head_dim == 0 || cols != heads * head_dim || head_dim % 2 != 0 {
        return Err(InferError::Dimension(format!(
            "draft RoPE invalide x={:?}, heads={heads}, head_dim={head_dim}",
            x.shape()
        )));
    }
    let half = head_dim / 2;
    let mut out = x.data().to_vec();
    for row in 0..rows {
        let pos = (position_offset + row) as f32;
        for head in 0..heads {
            let base = row * cols + head * head_dim;
            for dim in 0..half {
                let exponent = (2 * dim) as f32 / head_dim as f32;
                let angle = pos / theta.powf(exponent);
                let (sin, cos) = angle.sin_cos();
                let left = x.data()[base + dim];
                let right = x.data()[base + half + dim];
                out[base + dim] = left * cos - right * sin;
                out[base + half + dim] = right * cos + left * sin;
            }
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

fn attention_position_allowed(
    query_position: usize,
    key_position: usize,
    sliding_window: Option<usize>,
) -> bool {
    match sliding_window {
        Some(window) => query_position >= key_position && query_position < key_position + window,
        None => true,
    }
}

fn softmax_scores(scores: &[f32]) -> Result<Vec<f32>> {
    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return Err(InferError::Dimension(
            "draft attention sans position visible".to_string(),
        ));
    }
    let mut sum = 0.0_f32;
    let mut out = Vec::with_capacity(scores.len());
    for score in scores {
        let value = if score.is_finite() {
            (*score - max).exp()
        } else {
            0.0
        };
        sum += value;
        out.push(value);
    }
    if sum <= 0.0 || !sum.is_finite() {
        return Err(InferError::Dimension(
            "draft attention softmax invalide".to_string(),
        ));
    }
    for value in &mut out {
        *value /= sum;
    }
    Ok(out)
}

fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn accumulate_scaled(target: &mut [f32], source: &[f32], scale: f32) {
    for (left, right) in target.iter_mut().zip(source) {
        *left += right * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuantConfig;
    use safetensors::{
        serialize,
        tensor::{Dtype, View},
    };
    use std::borrow::Cow;
    use std::collections::HashMap;

    fn target_config() -> ModelConfig {
        ModelConfig {
            model_type: "qwen3_5_moe".to_string(),
            hidden_size: 2048,
            num_hidden_layers: 40,
            num_attention_heads: 16,
            num_key_value_heads: 2,
            num_global_key_value_heads: None,
            head_dim: Some(256),
            global_head_dim: None,
            intermediate_size: 0,
            rms_norm_eps: 1e-6,
            rope_theta: 10_000_000.0,
            vocab_size: 248_320,
            eos_token_ids: Vec::new(),
            tie_word_embeddings: false,
            quantization: Option::<QuantConfig>::None,
            full_attention_interval: None,
            attn_output_gate: None,
            partial_rotary_factor: None,
            linear_num_value_heads: None,
            linear_num_key_heads: None,
            linear_key_head_dim: None,
            linear_value_head_dim: None,
            linear_conv_kernel_dim: None,
            num_experts: None,
            num_experts_per_tok: None,
            top_k_experts: None,
            moe_intermediate_size: None,
            shared_expert_intermediate_size: None,
            mtp_num_hidden_layers: None,
            hidden_activation: None,
            hidden_act: None,
            rope_local_base_freq: None,
            rope_full_base_freq: None,
            rope_full_partial_rotary_factor: None,
            rope_sliding_partial_rotary_factor: None,
            sliding_window: None,
            sliding_window_pattern: None,
            layer_types: Vec::new(),
            attention_k_eq_v: false,
            enable_moe_block: false,
            query_pre_attn_scalar: None,
            attn_logit_softcapping: None,
            final_logit_softcapping: None,
            rope_scaling: None,
        }
    }

    #[test]
    fn validates_dflash_draft_config_against_target() {
        let temp = tempfile::tempdir().expect("invariant: tempdir");
        let config = r#"{
            "architectures":["DFlashDraftModel"],
            "hidden_size":2048,
            "num_hidden_layers":6,
            "num_target_layers":40,
            "intermediate_size":6144,
            "num_attention_heads":32,
            "num_key_value_heads":8,
            "head_dim":128,
            "rms_norm_eps":0.000001,
            "rope_parameters":{"rope_theta":10000000.0},
            "dflash_config":{
                "block_size":16,
                "mask_token_id":248077,
                "target_layer_ids":[1,6,11,16,22,27,32,37]
            }
        }"#;
        std::fs::write(temp.path().join("config.json"), config).expect("invariant: write config");
        let mut tensors = HashMap::new();
        insert_f32(&mut tensors, "fc.weight", &[1, 4]);
        let buffer = serialize(tensors, None).expect("invariant: serialize safetensors");
        std::fs::write(temp.path().join("model.safetensors"), buffer)
            .expect("invariant: write safetensors");

        let info = load_dflash_draft_for_target(&target_config(), temp.path())
            .expect("invariant: draft dflash valide");

        assert_eq!(info.block_size, 16);
        assert_eq!(info.mask_token_id, 248_077);
        assert_eq!(info.target_layer_ids, vec![1, 6, 11, 16, 22, 27, 32, 37]);
        assert_eq!(info.intermediate_size, 6144);
        assert_eq!(info.num_attention_heads, 32);
        assert_eq!(info.num_key_value_heads, 8);
        assert_eq!(info.head_dim, 128);
        assert_eq!(info.rms_norm_eps, 1e-6);
        assert_eq!(info.rope_theta, 10_000_000.0);
        assert_eq!(info.tensor_count, 1);
    }

    #[test]
    fn loads_typed_dflash_draft_weights() {
        let temp = tempfile::tempdir().expect("invariant: tempdir");
        let config = r#"{
            "architectures":["DFlashDraftModel"],
            "hidden_size":4,
            "num_hidden_layers":1,
            "num_target_layers":4,
            "intermediate_size":8,
            "num_attention_heads":2,
            "num_key_value_heads":1,
            "head_dim":2,
            "rms_norm_eps":0.000001,
            "rope_theta":10000.0,
            "sliding_window":8,
            "layer_types":["sliding_attention"],
            "dflash_config":{
                "block_size":4,
                "mask_token_id":15,
                "target_layer_ids":[0,2]
            }
        }"#;
        std::fs::write(temp.path().join("config.json"), config).expect("invariant: write config");
        let mut tensors = HashMap::new();
        insert_f32(&mut tensors, "hidden_norm.weight", &[4]);
        insert_f32(&mut tensors, "fc.weight", &[4, 8]);
        insert_f32(&mut tensors, "layers.0.input_layernorm.weight", &[4]);
        insert_f32(
            &mut tensors,
            "layers.0.post_attention_layernorm.weight",
            &[4],
        );
        insert_f32(&mut tensors, "layers.0.self_attn.q_proj.weight", &[4, 4]);
        insert_f32(&mut tensors, "layers.0.self_attn.k_proj.weight", &[2, 4]);
        insert_f32(&mut tensors, "layers.0.self_attn.v_proj.weight", &[2, 4]);
        insert_f32(&mut tensors, "layers.0.self_attn.o_proj.weight", &[4, 4]);
        insert_f32(&mut tensors, "layers.0.self_attn.q_norm.weight", &[2]);
        insert_f32(&mut tensors, "layers.0.self_attn.k_norm.weight", &[2]);
        insert_f32(&mut tensors, "layers.0.mlp.gate_proj.weight", &[8, 4]);
        insert_f32(&mut tensors, "layers.0.mlp.up_proj.weight", &[8, 4]);
        insert_f32(&mut tensors, "layers.0.mlp.down_proj.weight", &[4, 8]);
        insert_f32(&mut tensors, "norm.weight", &[4]);
        let buffer = serialize(tensors, None).expect("invariant: serialize safetensors");
        std::fs::write(temp.path().join("model.safetensors"), buffer)
            .expect("invariant: write safetensors");

        let draft = load_dflash_draft_weights_for_target(&tiny_target_config(), temp.path())
            .expect("invariant: poids draft dflash valides");

        assert_eq!(draft.info.hidden_size, 4);
        assert_eq!(draft.info.target_layer_ids, vec![0, 2]);
        assert_eq!(draft.hidden_norm.shape(), &[4]);
        assert_eq!(draft.fc.weight().shape(), &[4, 8]);
        assert_eq!(draft.layers.len(), 1);
        assert_eq!(draft.layers[0].attention.q_proj.weight().shape(), &[4, 4]);
        assert_eq!(draft.layers[0].attention.k_proj.weight().shape(), &[2, 4]);
        assert_eq!(draft.layers[0].attention.v_proj.weight().shape(), &[2, 4]);
        assert_eq!(draft.layers[0].attention.o_proj.weight().shape(), &[4, 4]);
        let (gate, up, down) = draft.layers[0].mlp.projections();
        assert_eq!(gate.weight().shape(), &[8, 4]);
        assert_eq!(up.weight().shape(), &[8, 4]);
        assert_eq!(down.weight().shape(), &[4, 8]);
        assert_eq!(draft.norm.shape(), &[4]);

        let target_hidden =
            Tensor::from_vec(vec![2, 8], (0..16).map(|idx| idx as f32 * 0.02).collect())
                .expect("invariant: target hidden valide");
        let projected = draft
            .project_target_hidden(&target_hidden, ForwardRuntime::cpu())
            .expect("invariant: projection contexte DFlash valide");
        assert_eq!(projected.shape(), &[2, 4]);

        let noise_embedding =
            Tensor::from_vec(vec![3, 4], (0..12).map(|idx| idx as f32 * 0.03).collect())
                .expect("invariant: noise embedding valide");
        let decoded = draft
            .forward_projected_context(&noise_embedding, &projected, ForwardRuntime::cpu())
            .expect("invariant: forward DFlash CPU valide");
        assert_eq!(decoded.shape(), &[3, 4]);
        assert!(decoded.data().iter().all(|value| value.is_finite()));
    }

    fn tiny_target_config() -> ModelConfig {
        let mut config = target_config();
        config.hidden_size = 4;
        config.num_hidden_layers = 4;
        config.num_attention_heads = 2;
        config.num_key_value_heads = 1;
        config.head_dim = Some(2);
        config.vocab_size = 16;
        config
    }

    fn insert_f32(tensors: &mut HashMap<String, F32View>, key: &str, shape: &[usize]) {
        let len = shape.iter().product::<usize>();
        let data = (0..len)
            .map(|idx| idx as f32 * 0.01)
            .flat_map(f32::to_le_bytes)
            .collect();
        tensors.insert(
            key.to_string(),
            F32View {
                shape: shape.to_vec(),
                data,
            },
        );
    }

    struct F32View {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for F32View {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }
}
