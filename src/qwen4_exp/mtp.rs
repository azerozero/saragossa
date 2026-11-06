//! Tête MTP de référence (CPU f32) de `qwen4_exp`.
//!
//! Absente de transformers : transcription de `graph_mtp` (llama.cpp,
//! `qwen4exp.cpp:580-674`). Pour des paires `(h_t, tok_{t+1})`, `h_t` étant le
//! résiduel hc du tronc avant le mixer final (`qwen4exp.cpp:555`) :
//!
//! - `e = RMSNorm(embed(tok_{t+1}))·(1 + w_e)`, recopié sur les `hc` flux ;
//! - `h_s = RMSNorm_flux(h_t)·(1 + w_h)`, normé flux par flux ;
//! - `res_s = fc_embedding·e + fc_hidden·h_s` (forme découpée de
//!   `eh_proj(concat(e, h_s))`, `qwen4exp.cpp:644`) ;
//! - une couche QSA + MoE à hyper-connections, sans PLE ;
//! - sortie hc (le `h` du draft suivant) puis mixer de tête et lm_head partagé.
//!
//! Le convertisseur plie chaque gain en `1 + w` (`qwen4exp.cpp:361`) ; nos
//! poids gardent `w`, et `rms_norm_1p` ajoute 1 une seule fois. Les lignes de
//! l'historique portent la position RoPE de leur indice : la RoPE étant
//! relative, un décalage constant de position ne change l'attention et les
//! scores de l'indexeur qu'à l'arrondi près.

use super::forward::{LayerOutput, Qwen4ExpReference};
use super::math::{linear, rms_norm_1p};
use crate::qwen_loader::qwen4_exp::Qwen4ExpLayerKind;
use crate::{InferError, Result, Tensor};

/// Préfixe de l'unique couche MTP.
const MTP_LAYER: &str = "mtp.layers.0.";

impl Qwen4ExpReference {
    /// Entrée de la couche MTP `[T, hc·hidden]` pour les paires
    /// `(h[t], next_ids[t])`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de forme, de vocabulaire ou de lecture.
    pub(crate) fn mtp_input(&self, h: &Tensor, next_ids: &[usize]) -> Result<Tensor> {
        let c = &self.checkpoint().config;
        let (tokens, width) = h.as_matrix()?;
        if tokens != next_ids.len() || width != c.hc_width() {
            return Err(InferError::Shape(format!(
                "MTP : h [{tokens}, {width}] pour {} tokens (largeur {})",
                next_ids.len(),
                c.hc_width()
            )));
        }
        let ckpt = self.checkpoint();
        let e = rms_norm_1p(
            self.embed_rows(next_ids)?.data(),
            c.hidden,
            c.hidden,
            &ckpt.floats("mtp.pre_fc_norm_embedding.weight")?,
            c.rms_eps,
        );
        let h_norm = rms_norm_1p(
            h.data(),
            width,
            c.hidden,
            &ckpt.floats("mtp.pre_fc_norm_hidden.weight")?,
            c.rms_eps,
        );
        let fe = linear(
            &Tensor::from_vec(vec![tokens, c.hidden], e)?,
            &ckpt.affine("mtp.fc_embedding")?,
        )?;
        // Flux à plat : une ligne de `hidden` par (token, flux).
        let fh = linear(
            &Tensor::from_vec(vec![tokens * c.hc_count, c.hidden], h_norm)?,
            &ckpt.affine("mtp.fc_hidden")?,
        )?;
        let mut out = fh.data().to_vec();
        for (t, row) in out.chunks_exact_mut(width).enumerate() {
            let e_row = &fe.data()[t * c.hidden..(t + 1) * c.hidden];
            for stream in row.chunks_exact_mut(c.hidden) {
                stream.iter_mut().zip(e_row).for_each(|(v, e)| *v += e);
            }
        }
        Tensor::from_vec(vec![tokens, width], out)
    }

    /// Couche MTP (QSA + MoE à hyper-connections) sur l'entrée `x`
    /// `[T, hc·hidden]`, sélection et experts imposables (oracles).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de forme ou si une imposition est invalide.
    pub(crate) fn mtp_layer(
        &self,
        x: &Tensor,
        forced: Option<&[Vec<usize>]>,
        forced_routing: Option<&[Option<Vec<usize>>]>,
    ) -> Result<LayerOutput> {
        self.layer_at(
            MTP_LAYER,
            Qwen4ExpLayerKind::Full,
            None,
            x,
            forced,
            forced_routing,
        )
    }

    /// Mixer de tête MTP (sans injection) : `[T, hidden]`, entrée du lm_head
    /// partagé.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture ou de forme.
    pub(crate) fn mtp_head(&self, h: &Tensor) -> Result<Tensor> {
        let c = &self.checkpoint().config;
        let mixer = self.residual("mtp.hyper_connection_mixer.", false)?;
        Ok(mixer.mix(h, c.hc_count, c.hidden, c.rms_eps)?.input)
    }
}
