//! Forward de référence CPU f32 `qwen4_exp`, couche par couche, sans cache.
//!
//! Chaque couche est lue, exécutée puis libérée : poids denses déquantifiés en
//! f32 pour la seule couche courante, experts déquantifiés un par un, lm_head
//! par blocs de lignes et embedding limité aux lignes utiles.

use super::attention::{qsa, QsaDims, QsaWeights};
use super::math::linear;
use super::moe::{moe, MoeWeights};
use super::ngram::ngram_ids;
use super::ple::{ple, PleWeights};
use super::residual::{combine, GatedResidual};
use crate::linear_attention::{LinearAttention, LinearAttentionConfig, LinearAttentionWeights};
use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpLayerKind};
use crate::{InferError, Linear, Result, Tensor};

/// Lignes de lm_head déquantifiées par bloc (≈ 256 Mo f32 à hidden 2560).
const LM_HEAD_BLOCK_ROWS: usize = 25_600;

/// Sortie d'une couche et traces de sélection.
pub struct LayerOutput {
    /// Résiduel élargi `[T, hc·hidden]` après la couche.
    pub hidden: Tensor,
    /// Experts routés par token (ensembles triés).
    pub routing: Vec<Vec<usize>>,
    /// Écart de probabilité au seuil du top-k du routeur, par token.
    pub routing_margins: Vec<f32>,
    /// `(k-ième, (k+1)-ième)` experts au seuil du top-k, par token.
    pub routing_boundary: Vec<(usize, usize)>,
    /// Trace QSA (couches d'attention seulement).
    pub qsa: Option<QsaTrace>,
}

/// Sélection propre de l'indexeur (tokens triés) et scores de blocs par requête.
pub struct QsaTrace {
    pub selected: Vec<Vec<usize>>,
    pub scores: Vec<Vec<f32>>,
}

/// Référence CPU f32 sur un checkpoint q4e8.
pub struct Qwen4ExpReference {
    ckpt: Qwen4ExpCheckpoint,
}

fn vector(data: Vec<f32>) -> Result<Tensor> {
    Tensor::from_vec(vec![data.len()], data)
}

fn tensor(rows: usize, cols: usize, data: Vec<f32>) -> Result<Tensor> {
    Tensor::from_vec(vec![rows, cols], data)
}

impl Qwen4ExpReference {
    /// Enveloppe un checkpoint ouvert.
    pub fn new(ckpt: Qwen4ExpCheckpoint) -> Self {
        Self { ckpt }
    }

    /// Nombre de blocs retenus par l'indexeur QSA (`budget / compress`).
    pub fn budget_blocks(&self) -> usize {
        let c = &self.ckpt.config;
        c.indexer_budget / c.indexer_compress
    }

    /// Taille des blocs de l'indexeur QSA.
    pub fn compress(&self) -> usize {
        self.ckpt.config.indexer_compress
    }

    /// Checkpoint sous-jacent.
    pub fn checkpoint(&self) -> &Qwen4ExpCheckpoint {
        &self.ckpt
    }

    /// Embedding répété sur les `hc` flux : `[T, hc·hidden]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un token sort du vocabulaire ou sur lecture.
    pub fn embed(&self, ids: &[usize]) -> Result<Tensor> {
        let c = &self.ckpt.config;
        let base = Qwen4ExpCheckpoint::trunk("embed_tokens");
        let mut out = Vec::with_capacity(ids.len() * c.hc_width());
        for &id in ids {
            if id >= c.vocab {
                return Err(InferError::Shape(format!("token {id} hors vocabulaire")));
            }
            let row = self.ckpt.affine_rows(&base, None, id..id + 1)?;
            for _ in 0..c.hc_count {
                out.extend_from_slice(row.data());
            }
        }
        tensor(ids.len(), c.hc_width(), out)
    }

    fn residual(&self, prefix: &str, inject: bool) -> Result<GatedResidual> {
        let c = &self.ckpt.config;
        Ok(GatedResidual {
            norm: self.ckpt.floats(&format!("{prefix}hc_norm.weight"))?,
            down: self
                .ckpt
                .affine(&format!("{prefix}input_mix_weight_down"))?,
            up: self.ckpt.affine(&format!("{prefix}input_mix_weight_up"))?,
            inject: if inject {
                Some(tensor(
                    c.hc_count,
                    c.hc_width(),
                    self.ckpt
                        .floats(&format!("{prefix}block_inject_weight.weight"))?,
                )?)
            } else {
                None
            },
        })
    }

    fn deltanet(&self, prefix: &str, x: &Tensor) -> Result<Tensor> {
        let c = &self.ckpt.config;
        let l = format!("{prefix}linear_attn.");
        let dense = |name: &str, rows: usize| -> Result<Linear> {
            Linear::new(
                tensor(rows, c.hidden, self.ckpt.floats(&format!("{l}{name}"))?)?,
                None,
            )
        };
        let conv_dim =
            2 * c.linear_key_heads * c.linear_key_dim + c.linear_value_heads * c.linear_value_dim;
        let weights = LinearAttentionWeights {
            in_proj_qkv: Linear::new(self.ckpt.affine(&format!("{l}in_proj_qkv"))?, None)?,
            in_proj_z: Linear::new(self.ckpt.affine(&format!("{l}in_proj_z"))?, None)?,
            in_proj_b: dense("in_proj_b.weight", c.linear_value_heads)?,
            in_proj_a: dense("in_proj_a.weight", c.linear_value_heads)?,
            out_proj: Linear::new(self.ckpt.affine(&format!("{l}out_proj"))?, None)?,
            conv1d_weight: Tensor::from_vec(
                vec![conv_dim, 1, c.conv_kernel],
                self.ckpt.floats(&format!("{l}conv1d.weight"))?,
            )?,
            a_log: vector(self.ckpt.floats(&format!("{l}A_log"))?)?,
            dt_bias: vector(self.ckpt.floats(&format!("{l}dt_bias"))?)?,
            // Gain direct `w` (Qwen4ExpTextRMSNormGated), sans ajout de 1.
            norm_weight: vector(self.ckpt.floats(&format!("{l}norm.weight"))?)?,
        };
        let config = LinearAttentionConfig {
            num_key_heads: c.linear_key_heads,
            num_value_heads: c.linear_value_heads,
            key_head_dim: c.linear_key_dim,
            value_head_dim: c.linear_value_dim,
            conv_kernel_dim: c.conv_kernel,
            rms_eps: c.rms_eps,
        };
        LinearAttention::new(weights).forward_flash_next_cpu(config, x, c.sigmoid_output_gate)
    }

    /// Attention QSA seule de la couche `index` sur `x` `[T, hidden]` (oracle).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de forme ou si la couche n'est pas QSA.
    #[cfg(test)]
    pub(crate) fn attention_layer(
        &self,
        index: usize,
        x: &Tensor,
        forced: Option<&[Vec<usize>]>,
    ) -> Result<(Tensor, QsaTrace)> {
        if self.ckpt.config.layer_kinds.get(index) != Some(&Qwen4ExpLayerKind::Full) {
            return Err(InferError::Config(format!(
                "couche {index}: pas d'attention QSA"
            )));
        }
        self.attention(&Qwen4ExpCheckpoint::layer_prefix(index), x, forced)
    }

    fn attention(
        &self,
        prefix: &str,
        x: &Tensor,
        forced: Option<&[Vec<usize>]>,
    ) -> Result<(Tensor, QsaTrace)> {
        let c = &self.ckpt.config;
        let a = format!("{prefix}self_attn.");
        let w = QsaWeights {
            q_proj: self.ckpt.affine(&format!("{a}q_proj"))?,
            k_proj: self.ckpt.affine(&format!("{a}k_proj"))?,
            v_proj: self.ckpt.affine(&format!("{a}v_proj"))?,
            o_proj: self.ckpt.affine(&format!("{a}o_proj"))?,
            q_norm: self.ckpt.floats(&format!("{a}q_norm.weight"))?,
            k_norm: self.ckpt.floats(&format!("{a}k_norm.weight"))?,
            index_qk: self.ckpt.affine(&format!("{a}indexer.index_qk_proj"))?,
            index_q_norm: self
                .ckpt
                .floats(&format!("{a}indexer.q_layernorm.weight"))?,
            index_k_norm: self
                .ckpt
                .floats(&format!("{a}indexer.k_layernorm.weight"))?,
        };
        let d = QsaDims {
            heads: c.heads,
            kv_heads: c.kv_heads,
            head_dim: c.head_dim,
            rotary_dim: c.rotary_dim,
            rope_theta: c.rope_theta,
            index_heads: c.indexer_heads,
            index_head_dim: c.indexer_head_dim,
            budget: c.indexer_budget,
            compress: c.indexer_compress,
            eps: c.rms_eps,
        };
        let out = qsa(x, &w, d, forced)?;
        Ok((
            out.output,
            QsaTrace {
                selected: out.selected,
                scores: out.scores,
            },
        ))
    }

    fn ple_contribution(&self, prefix: &str, h: &Tensor, ids: &[usize]) -> Result<Tensor> {
        let c = &self.ckpt.config;
        let p = &self.ckpt.ngram_params;
        let tokens: Vec<u64> = ids.iter().map(|&t| t as u64).collect();
        let rows = ngram_ids(
            &tokens,
            c.eos,
            c.ngram_size,
            c.heads_per_ngram,
            &p.multipliers,
            &p.vocab_sizes,
            &p.offsets,
        );
        let embeddings = tensor(ids.len(), c.ple_dim, self.ckpt.ngram_rows(&rows)?)?;
        let e = format!("{prefix}ple.");
        let w = PleWeights {
            key_proj: self.ckpt.affine(&format!("{e}key_proj"))?,
            value_proj: self.ckpt.affine(&format!("{e}value_proj"))?,
            norm_key: self.ckpt.floats(&format!("{e}norm_key.weight"))?,
            norm_query: self.ckpt.floats(&format!("{e}norm_query.weight"))?,
            norm_conv: self.ckpt.floats(&format!("{e}norm_conv.weight"))?,
            conv: self.ckpt.floats(&format!("{e}conv1d.weight"))?,
            kernel: c.ple_conv_kernel,
            dilation: c.ngram_size,
        };
        ple(h, &embeddings, &w, c.hc_count, c.hidden, c.rms_eps)
    }

    fn moe_block(
        &self,
        prefix: &str,
        x: &Tensor,
        forced: Option<&[Option<Vec<usize>>]>,
    ) -> Result<super::moe::MoeOutput> {
        let c = &self.ckpt.config;
        let m = format!("{prefix}mlp.");
        let w = MoeWeights {
            router: tensor(
                c.experts,
                c.hidden,
                self.ckpt.floats(&format!("{m}gate.weight"))?,
            )?,
            shared_gate: self.ckpt.affine(&format!("{m}shared_expert.gate_proj"))?,
            shared_up: self.ckpt.affine(&format!("{m}shared_expert.up_proj"))?,
            shared_down: self.ckpt.affine(&format!("{m}shared_expert.down_proj"))?,
            shared_expert_gate: tensor(
                1,
                c.hidden,
                self.ckpt.floats(&format!("{m}shared_expert_gate.weight"))?,
            )?,
            top_k: c.top_k,
            norm_topk_prob: c.norm_topk_prob,
        };
        let out = moe(x, &w, forced, |e| {
            Ok((
                self.ckpt.expert(&format!("{m}switch_mlp.gate_proj"), e)?,
                self.ckpt.expert(&format!("{m}switch_mlp.up_proj"), e)?,
                self.ckpt.expert(&format!("{m}switch_mlp.down_proj"), e)?,
            ))
        })?;
        Ok(out)
    }

    /// Une couche complète sur `h` (`[T, hc·hidden]`), poids lus puis libérés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture ou de forme.
    pub fn layer(&self, index: usize, h: &Tensor, ids: &[usize]) -> Result<LayerOutput> {
        self.layer_with_selection(index, h, ids, None)
    }

    /// Comme [`Self::layer`], avec une sélection QSA imposée (oracle).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de forme ou si la sélection n'est pas causale.
    pub fn layer_with_selection(
        &self,
        index: usize,
        h: &Tensor,
        ids: &[usize],
        forced: Option<&[Vec<usize>]>,
    ) -> Result<LayerOutput> {
        self.layer_forced(index, h, ids, forced, None)
    }

    /// Comme [`Self::layer_with_selection`], avec en plus des experts imposés
    /// par token (`Some(ensemble)` aux quasi-égalités de l'oracle).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de forme ou si la sélection n'est pas causale.
    pub(crate) fn layer_forced(
        &self,
        index: usize,
        h: &Tensor,
        ids: &[usize],
        forced: Option<&[Vec<usize>]>,
        forced_routing: Option<&[Option<Vec<usize>>]>,
    ) -> Result<LayerOutput> {
        let c = &self.ckpt.config;
        let kind = *c
            .layer_kinds
            .get(index)
            .ok_or_else(|| InferError::Shape(format!("couche {index} hors modèle")))?;
        let prefix = Qwen4ExpCheckpoint::layer_prefix(index);
        let mut h = h.clone();
        if index == c.ple_layer {
            // PLE ajouté aux flux AVANT l'hyper-connection d'attention.
            h = h.add(&self.ple_contribution(&prefix, &h, ids)?)?;
        }
        let attn_hc = self.residual(&format!("{prefix}attn_hyper_connection."), true)?;
        let mixed = attn_hc.mix(&h, c.hc_count, c.hidden, c.rms_eps)?;
        let (branch, selected) = match kind {
            Qwen4ExpLayerKind::Linear => (self.deltanet(&prefix, &mixed.input)?, None),
            Qwen4ExpLayerKind::Full => {
                let (out, trace) = self.attention(&prefix, &mixed.input, forced)?;
                (out, Some(trace))
            }
        };
        let inject = mixed
            .inject
            .ok_or_else(|| InferError::Config("hyper-connection sans injection".to_string()))?;
        let h = combine(&h, &branch, &inject, c.hc_count)?;
        let mlp_hc = self.residual(&format!("{prefix}mlp_hyper_connection."), true)?;
        let mixed = mlp_hc.mix(&h, c.hc_count, c.hidden, c.rms_eps)?;
        let moe_out = self.moe_block(&prefix, &mixed.input, forced_routing)?;
        let inject = mixed
            .inject
            .ok_or_else(|| InferError::Config("hyper-connection sans injection".to_string()))?;
        Ok(LayerOutput {
            hidden: combine(&h, &moe_out.output, &inject, c.hc_count)?,
            routing: moe_out.routing,
            routing_margins: moe_out.margins,
            routing_boundary: moe_out.boundary,
            qsa: selected,
        })
    }

    /// Mixer final (sans injection ni norme supplémentaire) : `[T, hidden]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture ou de forme.
    pub fn final_mixer(&self, h: &Tensor) -> Result<Tensor> {
        let c = &self.ckpt.config;
        let mixer = self.residual(&Qwen4ExpCheckpoint::trunk("hyper_connection_mixer."), false)?;
        Ok(mixer.mix(h, c.hc_count, c.hidden, c.rms_eps)?.input)
    }

    /// Forward complet sans cache : embedding, couches, mixer final, logits.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture ou de forme.
    pub fn forward_logits(&self, ids: &[usize]) -> Result<Tensor> {
        let mut h = self.embed(ids)?;
        for index in 0..self.ckpt.config.layer_kinds.len() {
            h = self.layer(index, &h, ids)?.hidden;
        }
        let final_hidden = self.final_mixer(&h)?;
        self.logits(&final_hidden)
    }

    /// Logits `[T, vocab]`, lm_head déquantifié par blocs de lignes.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture ou de forme.
    pub fn logits(&self, final_hidden: &Tensor) -> Result<Tensor> {
        let c = &self.ckpt.config;
        let (tokens, _) = final_hidden.as_matrix()?;
        let mut out = vec![0.0_f32; tokens * c.vocab];
        let mut start = 0;
        while start < c.vocab {
            let end = (start + LM_HEAD_BLOCK_ROWS).min(c.vocab);
            let block = self
                .ckpt
                .affine_rows(Qwen4ExpCheckpoint::lm_head(), None, start..end)?;
            let part = linear(final_hidden, &block)?;
            for t in 0..tokens {
                out[t * c.vocab + start..t * c.vocab + end]
                    .copy_from_slice(&part.data()[t * (end - start)..(t + 1) * (end - start)]);
            }
            start = end;
        }
        tensor(tokens, c.vocab, out)
    }
}
