//! Validation et chargement des checkpoints draft DFlash.

use crate::{
    catalog::read_safetensors_keys, rms_norm, safetensor::bytes_to_dense_f32,
    AffineQuantizedTensor, ForwardRuntime, GatedMlp, InferError, Linear, ModelConfig, Result,
    Tensor,
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
    /// Taille du vocabulaire déclarée par le draft (codebooks du sélecteur).
    pub vocab_size: usize,
    /// DFlash 2 : taps de la convolution causale autour de chaque sous-couche (0 = DFlash 1).
    pub conv_kernel_size: usize,
    /// DFlash 2 : canaux partageant un noyau dynamique.
    pub conv_group_size: usize,
    /// DFlash 2 : rang de la transition du sélecteur de candidats.
    pub selector_rank: usize,
    /// DFlash 2 : candidats unaires reclassés par position.
    pub selector_top_k: usize,
}

/// Quantification affine appliquée aux projections du draft au chargement
/// (fc, q/k/v/o, gate/up/down, noyaux dynamiques). Normes, noyaux statiques,
/// sélecteur et codebooks restent en f32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DFlashQuant {
    pub group_size: usize,
    pub bits: usize,
}

impl DFlashQuant {
    /// `SARAGOSSA_RUST_DFLASH_QUANT_BITS` : absent ou `0` → dense f32 ; `4` ou `8` →
    /// affine par groupes de 64 (format des kernels u4/u8 gs64 du tronc).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la variable n'est ni vide, ni `0`, `4` ou `8`.
    pub fn from_env() -> Result<Option<Self>> {
        match crate::runtime_flags::env_var("SARAGOSSA_RUST_DFLASH_QUANT_BITS") {
            None => Ok(None),
            Some(value) => match value.trim() {
                "" | "0" => Ok(None),
                "4" => Ok(Some(Self {
                    group_size: 64,
                    bits: 4,
                })),
                "8" => Ok(Some(Self {
                    group_size: 64,
                    bits: 8,
                })),
                other => Err(InferError::Config(format!(
                    "SARAGOSSA_RUST_DFLASH_QUANT_BITS={other} : attendu 0, 4 ou 8"
                ))),
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct DFlashDraft {
    pub info: DFlashDraftInfo,
    /// Quantification appliquée aux projections (None : dense f32).
    pub quant: Option<DFlashQuant>,
    pub hidden_norm: Tensor,
    pub fc: Linear,
    pub layers: Vec<DFlashDraftLayer>,
    pub norm: Tensor,
    /// DFlash 2 : sélecteur de candidats (absent en DFlash 1 : argmax par position).
    pub selector: Option<DFlashSelector>,
}

/// DFlash 2 : convolution causale groupée à noyau dynamique autour d'une sous-couche.
/// `prepare` convole l'entrée (noyau 0), `finish` la sortie (noyau 1) avec le noyau
/// dynamique produit à l'entrée. Zéro-remplie au début du bloc : ne traverse jamais
/// la frontière ancre/bloc.
#[derive(Debug, Clone)]
pub struct DFlashConv {
    /// `[2, kernel_size, hidden]` : noyaux statiques d'entrée (0) et de sortie (1).
    pub base_kernel: Tensor,
    /// `[2 * kernel_size * groups, hidden]` : noyaux dynamiques par token.
    pub kernel_projection: Linear,
    pub kernel_size: usize,
    pub group_size: usize,
}

/// DFlash 2 : reclassement des `top_k` candidats unaires par un score de transition
/// de rang `rank` : `unaire + <pred[précédent] ⊙ proj(h), succ[candidat]>`.
#[derive(Debug, Clone)]
pub struct DFlashSelector {
    /// `[rank, hidden]`.
    pub hidden_projection: Linear,
    /// `[vocab, rank]`.
    pub predecessor_codebook: Tensor,
    /// `[vocab, rank]`.
    pub successor_codebook: Tensor,
    pub rank: usize,
    pub top_k: usize,
}

#[derive(Debug, Clone)]
pub struct DFlashDraftLayer {
    pub input_norm: Tensor,
    pub attention: DFlashAttentionWeights,
    pub post_attention_norm: Tensor,
    pub mlp: GatedMlp,
    pub attention_conv: Option<DFlashConv>,
    pub mlp_conv: Option<DFlashConv>,
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
        let (attn_in, attn_dynamic) = match &self.attention_conv {
            Some(conv) => conv.prepare(&normed, runtime)?,
            None => (normed, Vec::new()),
        };
        let mut attn_out =
            self.attention
                .forward(info, layer_idx, &attn_in, draft_context, runtime)?;
        if let Some(conv) = &self.attention_conv {
            attn_out = conv.finish(&attn_out, &attn_dynamic)?;
        }
        let attention_state = hidden_states.add(&attn_out)?;
        let mlp_normed = rms_norm(
            &attention_state,
            &self.post_attention_norm,
            info.rms_norm_eps,
        )?;
        let (mlp_input, mlp_dynamic) = match &self.mlp_conv {
            Some(conv) => conv.prepare(&mlp_normed, runtime)?,
            None => (mlp_normed, Vec::new()),
        };
        let mut mlp_out = self.mlp.forward_with_runtime(&mlp_input, runtime)?;
        if let Some(conv) = &self.mlp_conv {
            mlp_out = conv.finish(&mlp_out, &mlp_dynamic)?;
        }
        attention_state.add(&mlp_out)
    }
}

impl DFlashConv {
    fn groups(&self, hidden: usize) -> Result<usize> {
        if self.group_size == 0 || hidden % self.group_size != 0 {
            return Err(InferError::Dimension(format!(
                "draft DFlash conv group_size={} ne divise pas hidden={hidden}",
                self.group_size
            )));
        }
        Ok(hidden / self.group_size)
    }

    /// Noyau dynamique par token `[len, 2 * kernel_size * groups]` et entrée convolée.
    pub fn prepare(
        &self,
        hidden: &Tensor,
        runtime: ForwardRuntime<'_>,
    ) -> Result<(Tensor, Vec<f32>)> {
        let dynamic = self
            .kernel_projection
            .forward_with_runtime(hidden, runtime)?
            .into_data();
        let output = self.convolve(hidden, &dynamic, 0)?;
        Ok((output, dynamic))
    }

    /// Sortie de la sous-couche convolée avec le noyau dynamique de `prepare`.
    pub fn finish(&self, hidden: &Tensor, dynamic: &[f32]) -> Result<Tensor> {
        self.convolve(hidden, dynamic, 1)
    }

    /// `out[t, c] = Σ_off (base[which, off, c] + dyn[t, which, off, g(c)]) · x[t - off, c]`,
    /// termes `t - off < 0` ignorés (zéro-remplissage causal).
    fn convolve(&self, hidden: &Tensor, dynamic: &[f32], which: usize) -> Result<Tensor> {
        let (len, width) = hidden.as_matrix()?;
        let groups = self.groups(width)?;
        let per_token = 2 * self.kernel_size * groups;
        if dynamic.len() != len * per_token {
            return Err(InferError::Dimension(format!(
                "draft DFlash conv dynamique {} valeurs, attendu {}",
                dynamic.len(),
                len * per_token
            )));
        }
        expect_shape(
            self.base_kernel.shape(),
            &[2, self.kernel_size, width],
            "draft DFlash conv base_kernel",
        )?;
        let base = self.base_kernel.data();
        let x = hidden.data();
        let mut out = vec![0.0f32; len * width];
        for t in 0..len {
            for off in 0..self.kernel_size.min(t + 1) {
                let source = &x[(t - off) * width..(t - off + 1) * width];
                let base_row = &base[(which * self.kernel_size + off) * width..][..width];
                let dyn_row =
                    &dynamic[t * per_token + (which * self.kernel_size + off) * groups..][..groups];
                let target = &mut out[t * width..(t + 1) * width];
                for (c, value) in target.iter_mut().enumerate() {
                    *value += (base_row[c] + dyn_row[c / self.group_size]) * source[c];
                }
            }
        }
        Tensor::from_vec(vec![len, width], out)
    }
}

impl DFlashSelector {
    /// Candidats `(token, logit)` des `top_k` logits d'une position, ordre quelconque.
    pub fn top_candidates(&self, logits: &[f32]) -> Vec<(usize, f32)> {
        let k = self.top_k.min(logits.len());
        let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
        if k < indexed.len() {
            indexed.select_nth_unstable_by(k, |a, b| b.1.total_cmp(&a.1));
            indexed.truncate(k);
        }
        indexed
    }

    /// Parcourt le bloc depuis `anchor` : à chaque position le candidat de score
    /// maximal `unaire + <pred[précédent] ⊙ proj(h), succ[candidat]>` devient le
    /// précédent de la suivante. `hidden` : `[positions, hidden]`, `candidates[p]` :
    /// les candidats de la position `p`.
    pub fn select(
        &self,
        hidden: &Tensor,
        candidates: &[Vec<(usize, f32)>],
        anchor: usize,
        runtime: ForwardRuntime<'_>,
    ) -> Result<Vec<usize>> {
        let (positions, _) = hidden.as_matrix()?;
        if candidates.len() != positions {
            return Err(InferError::Dimension(format!(
                "draft DFlash sélecteur : {} positions, {} listes de candidats",
                positions,
                candidates.len()
            )));
        }
        let projected = self
            .hidden_projection
            .forward_with_runtime(hidden, runtime)?;
        self.select_projected(projected.data(), candidates, anchor)
    }

    /// Comme [`Self::select`], avec la projection `[positions, rank]` déjà calculée
    /// (chemin résident : projection sur GPU, parcours des codebooks sur CPU).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les tailles divergent ou si un candidat sort du
    /// vocabulaire.
    pub fn select_projected(
        &self,
        projected: &[f32],
        candidates: &[Vec<(usize, f32)>],
        anchor: usize,
    ) -> Result<Vec<usize>> {
        let positions = candidates.len();
        if projected.len() != positions * self.rank {
            return Err(InferError::Dimension(format!(
                "draft DFlash sélecteur : projection de {} valeurs, attendu {positions}×{}",
                projected.len(),
                self.rank
            )));
        }
        let vocab = self.predecessor_codebook.shape()[0];
        fn codebook_row(codebook: &Tensor, token: usize, vocab: usize) -> Result<&[f32]> {
            if token >= vocab {
                return Err(InferError::Dimension(format!(
                    "draft DFlash sélecteur token {token} hors vocabulaire {vocab}"
                )));
            }
            codebook.row_slice(token)
        }
        let mut previous = anchor;
        let mut path = Vec::with_capacity(positions);
        for (position, row_candidates) in candidates.iter().enumerate() {
            let proj = &projected[position * self.rank..(position + 1) * self.rank];
            let pred = codebook_row(&self.predecessor_codebook, previous, vocab)?;
            let gate: Vec<f32> = pred.iter().zip(proj).map(|(p, h)| p * h).collect();
            let mut best: Option<(usize, f32)> = None;
            for &(token, unary) in row_candidates {
                let score =
                    unary + dot(&gate, codebook_row(&self.successor_codebook, token, vocab)?);
                if best.is_none_or(|(_, b)| score > b) {
                    best = Some((token, score));
                }
            }
            let (token, _) = best.ok_or_else(|| {
                InferError::Config("draft DFlash sélecteur sans candidat".to_string())
            })?;
            path.push(token);
            previous = token;
        }
        Ok(path)
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
                for (key_pos, &weight) in weights.iter().take(ctx_len).enumerate() {
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
    vocab_size: Option<usize>,
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
    #[serde(default)]
    conv_kernel_size: Option<usize>,
    #[serde(default)]
    conv_group_size: Option<usize>,
    #[serde(default)]
    selector_rank: Option<usize>,
    #[serde(default)]
    selector_top_k: Option<usize>,
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

    let is_v2 = raw
        .architectures
        .iter()
        .any(|name| name == "DFlash2DraftModel");
    if !is_v2
        && !raw
            .architectures
            .iter()
            .any(|name| name == "DFlashDraftModel")
    {
        return Err(InferError::Config(format!(
            "draft DFlash architecture absente: {:?}",
            raw.architectures
        )));
    }
    let v2 = |value: Option<usize>, name: &str| -> Result<usize> {
        match (is_v2, value) {
            (false, _) => Ok(0),
            (true, Some(value)) if value > 0 => Ok(value),
            (true, _) => Err(InferError::Config(format!(
                "draft DFlash 2 requiert dflash_config.{name} > 0"
            ))),
        }
    };
    let conv_kernel_size = v2(raw.dflash_config.conv_kernel_size, "conv_kernel_size")?;
    let conv_group_size = v2(raw.dflash_config.conv_group_size, "conv_group_size")?;
    let selector_rank = v2(raw.dflash_config.selector_rank, "selector_rank")?;
    let selector_top_k = v2(raw.dflash_config.selector_top_k, "selector_top_k")?;
    if is_v2 && raw.hidden_size % conv_group_size != 0 {
        return Err(InferError::Config(format!(
            "draft DFlash 2 conv_group_size={conv_group_size} ne divise pas hidden_size={}",
            raw.hidden_size
        )));
    }
    let vocab_size = match (is_v2, raw.vocab_size) {
        (true, Some(vocab)) if vocab > 0 => vocab,
        (true, _) => {
            return Err(InferError::Config(
                "draft DFlash 2 requiert vocab_size > 0".to_string(),
            ))
        }
        (false, vocab) => vocab.unwrap_or(0),
    };
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
        vocab_size,
        conv_kernel_size,
        conv_group_size,
        selector_rank,
        selector_top_k,
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
    load_dflash_draft_weights_quantized(target, draft_dir, None)
}

/// Comme [`load_dflash_draft_weights_for_target`], avec les projections
/// quantifiées en affine au chargement si `quant` est fourni.
///
/// # Errors
///
/// Renvoie une erreur si un poids requis est absent, si une forme diverge de la
/// config draft ou si une dimension d'entrée n'est pas un multiple du groupe.
pub fn load_dflash_draft_weights_quantized(
    target: &ModelConfig,
    draft_dir: impl AsRef<Path>,
    quant: Option<DFlashQuant>,
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
    let fc = load_dflash_linear(&st, "fc", hidden, fc_in, quant)?;
    let mut layers = Vec::with_capacity(info.num_hidden_layers);
    for layer_idx in 0..info.num_hidden_layers {
        let prefix = format!("layers.{layer_idx}");
        let attention_prefix = format!("{prefix}.self_attn");
        let mlp_prefix = format!("{prefix}.mlp");
        let attention = DFlashAttentionWeights {
            q_proj: load_dflash_linear(
                &st,
                &format!("{attention_prefix}.q_proj"),
                q_dim,
                hidden,
                quant,
            )?,
            k_proj: load_dflash_linear(
                &st,
                &format!("{attention_prefix}.k_proj"),
                kv_dim,
                hidden,
                quant,
            )?,
            v_proj: load_dflash_linear(
                &st,
                &format!("{attention_prefix}.v_proj"),
                kv_dim,
                hidden,
                quant,
            )?,
            o_proj: load_dflash_linear(
                &st,
                &format!("{attention_prefix}.o_proj"),
                hidden,
                q_dim,
                quant,
            )?,
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
                quant,
            )?,
            load_dflash_linear(
                &st,
                &format!("{mlp_prefix}.up_proj"),
                info.intermediate_size,
                hidden,
                quant,
            )?,
            load_dflash_linear(
                &st,
                &format!("{mlp_prefix}.down_proj"),
                hidden,
                info.intermediate_size,
                quant,
            )?,
        );
        let conv = |name: &str| -> Result<Option<DFlashConv>> {
            if info.conv_kernel_size == 0 {
                return Ok(None);
            }
            let groups = hidden / info.conv_group_size;
            let base_kernel = load_dflash_tensor(&st, &format!("{prefix}.{name}.base_kernel"))?;
            expect_shape(
                base_kernel.shape(),
                &[2, info.conv_kernel_size, hidden],
                &format!("{prefix}.{name}.base_kernel"),
            )?;
            Ok(Some(DFlashConv {
                base_kernel,
                kernel_projection: load_dflash_linear(
                    &st,
                    &format!("{prefix}.{name}.kernel_projection"),
                    2 * info.conv_kernel_size * groups,
                    hidden,
                    quant,
                )?,
                kernel_size: info.conv_kernel_size,
                group_size: info.conv_group_size,
            }))
        };
        let attention_conv = conv("attention_conv")?;
        let mlp_conv = conv("mlp_conv")?;
        layers.push(DFlashDraftLayer {
            input_norm: load_dflash_norm(&st, &format!("{prefix}.input_layernorm.weight"), hidden)?,
            attention,
            post_attention_norm: load_dflash_norm(
                &st,
                &format!("{prefix}.post_attention_layernorm.weight"),
                hidden,
            )?,
            mlp,
            attention_conv,
            mlp_conv,
        });
    }
    let norm = load_dflash_norm(&st, "norm.weight", hidden)?;
    let selector = if info.selector_rank > 0 {
        let codebook = |key: &str| -> Result<Tensor> {
            let tensor = load_dflash_tensor(&st, key)?;
            expect_shape(tensor.shape(), &[info.vocab_size, info.selector_rank], key)?;
            Ok(tensor)
        };
        Some(DFlashSelector {
            hidden_projection: load_dflash_linear(
                &st,
                "candidate_selector.hidden_projection",
                info.selector_rank,
                hidden,
                None,
            )?,
            predecessor_codebook: codebook("candidate_selector.predecessor_codebook")?,
            successor_codebook: codebook("candidate_selector.successor_codebook")?,
            rank: info.selector_rank,
            top_k: info.selector_top_k,
        })
    } else {
        None
    };
    Ok(DFlashDraft {
        info,
        quant,
        hidden_norm,
        fc,
        layers,
        norm,
        selector,
    })
}

fn load_dflash_linear(
    st: &SafeTensors,
    prefix: &str,
    expected_rows: usize,
    expected_cols: usize,
    quant: Option<DFlashQuant>,
) -> Result<Linear> {
    let key = format!("{prefix}.weight");
    let tensor = load_dflash_tensor(st, &key)?;
    expect_shape(tensor.shape(), &[expected_rows, expected_cols], &key)?;
    match quant {
        Some(DFlashQuant { group_size, bits }) => Linear::new_quantized(
            AffineQuantizedTensor::quantize(&tensor, group_size, bits)?,
            None,
        ),
        None => Linear::new(tensor, None),
    }
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
            output_gate_type: None,
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

#[cfg(test)]
mod dflash2_tests {
    use super::{DFlashConv, DFlashSelector};
    use crate::{ForwardRuntime, Linear, Tensor};

    #[test]
    fn conv_is_causal_grouped_and_dynamic() {
        // hidden 4, groupes de 2, 2 taps. Projection nulle sauf pour (noyau 0, tap 1, groupe 1)
        // qui vaut la somme des entrées du token : noyau dynamique = Σx.
        let hidden =
            Tensor::from_vec(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
        // base[0] : tap 0 = identité, tap 1 = 0 ; base[1] : tap 0 = 2, tap 1 = 0.
        let base = Tensor::from_vec(
            vec![2, 2, 4],
            vec![
                1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0,
            ],
        )
        .unwrap();
        // Lignes de la projection : [noyau, tap, groupe] = (0,0,0) (0,0,1) (0,1,0) (0,1,1) (1,0,0) (1,0,1) (1,1,0) (1,1,1).
        let mut proj = vec![0.0; 8 * 4];
        proj[3 * 4..4 * 4].copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let conv = DFlashConv {
            base_kernel: base,
            kernel_projection: Linear::new(Tensor::from_vec(vec![8, 4], proj).unwrap(), None)
                .unwrap(),
            kernel_size: 2,
            group_size: 2,
        };
        let (out, dynamic) = conv.prepare(&hidden, ForwardRuntime::cpu()).unwrap();
        // Token 0 : identité (pas de tap 1). Token 1 : identité + tap 1 dynamique (Σx = 4) sur le groupe 1
        // appliqué à x[0] = [.., 3, 4].
        assert_eq!(
            out.data(),
            &[
                1.0,
                2.0,
                3.0,
                4.0,
                1.0,
                1.0,
                1.0 + 4.0 * 3.0,
                1.0 + 4.0 * 4.0
            ]
        );
        let finished = conv.finish(&hidden, &dynamic).unwrap();
        assert_eq!(finished.data(), &[2.0, 4.0, 6.0, 8.0, 2.0, 2.0, 2.0, 2.0]);
    }

    #[test]
    fn selector_walks_from_anchor_with_transition_scores() {
        // vocab 3, rang 1. pred = [1, 2, 3], succ = [1, 0, -1], projection = identité sur hidden 1.
        let selector = DFlashSelector {
            hidden_projection: Linear::new(Tensor::from_vec(vec![1, 1], vec![1.0]).unwrap(), None)
                .unwrap(),
            predecessor_codebook: Tensor::from_vec(vec![3, 1], vec![1.0, 2.0, 3.0]).unwrap(),
            successor_codebook: Tensor::from_vec(vec![3, 1], vec![1.0, 0.0, -1.0]).unwrap(),
            rank: 1,
            top_k: 2,
        };
        let candidates = selector.top_candidates(&[0.5, 0.1, 0.4]);
        assert_eq!(candidates.len(), 2);
        assert!(candidates.iter().any(|&(t, _)| t == 0) && candidates.iter().any(|&(t, _)| t == 2));
        // Position 0 (h = 1, précédent = ancre 1, pred 2) : token 0 → 0.5 + 2·1 = 2.5 ; token 2 → 0.4 − 2 = −1.6.
        // Position 1 (h = −1, précédent 0, pred 1) : token 0 → 0.5 − 1 = −0.5 ; token 2 → 0.4 + 1 = 1.4.
        let hidden = Tensor::from_vec(vec![2, 1], vec![1.0, -1.0]).unwrap();
        let path = selector
            .select(
                &hidden,
                &[candidates.clone(), candidates],
                1,
                ForwardRuntime::cpu(),
            )
            .unwrap();
        assert_eq!(path, vec![0, 2]);
    }
}
