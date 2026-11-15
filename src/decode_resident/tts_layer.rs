//! Couche du code predictor Qwen3-TTS en 8 dispatches (trame échantillonnée).
//!
//! Le chemin général `encode_full_attn_dense_layer` encode 16 dispatches par
//! couche sur ce modèle (q, k et v séparés, norm+RoPE de Q puis de K, deux
//! copies K/V, gate, up, SwiGLU, down puis somme résiduelle). Ici : norme
//! d'entrée, QKV concaténé, norm+RoPE de Q et K avec append K/V, attention,
//! O, résiduel + post-norme, gate+up+SwiGLU, down + résiduel.
//!
//! NOTE: les normes en prologue du QKV et du gate+up (6 dispatches) ont été
//! mesurées plus lentes (+1,3 ms par trame) : chaque threadgroup recalcule la
//! norme de la ligne. Elles restent des dispatches séparés.
//!
//! Chaque kernel fusionné reprend la boucle et les associations du kernel
//! qu'il remplace : la sortie de couche est identique au bit à celle du chemin
//! général (`tts_cp_layer_matches_general_layer`). Seule la trame fusionnée
//! du code predictor l'emprunte ; le glouton et le decode LLM n'y passent pas.

use super::utils::byte_offset_f32;
use super::*;

/// Poids résolus d'une couche du code predictor pour [`DecodeResidentState::encode_tts_cp_layer`].
#[derive(Clone, Copy)]
pub(crate) struct TtsCpLayerWeights<'a> {
    pub rope_frequencies: &'a BufferRef,
    pub input_norm: &'a BufferRef,
    /// Projection q, k, v concaténée (lignes dans cet ordre).
    pub qkv_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a BufferRef,
    pub k_norm: &'a BufferRef,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub post_norm: &'a BufferRef,
    pub gate_proj: &'a MetalLinearWeightBuffers,
    pub up_proj: &'a MetalLinearWeightBuffers,
    pub down_proj: &'a MetalLinearWeightBuffers,
}

impl DecodeResidentState {
    /// Encode une couche du code predictor : `layer_in [hidden]` → `layer_out`.
    ///
    /// Couche Qwen3 sans gate de sortie, sans norme de V et sans sinks, poids
    /// u6 gs64 alignés : l'appelant le vérifie à la résolution des poids.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si un kernel fusionné refuse les poids (vérification
    /// manquée en amont), si le KV déborde ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow d'une couche : exécuteur + KV + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_tts_cp_layer(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        kv: &mut FullAttentionMetalState,
        weights: TtsCpLayerWeights<'_>,
        dims: FullAttnLayerDims,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        if dims.attn_output_gate || kv.has_sinks() {
            return Err(InferError::Config(
                "couche TTS fusionnée: gate de sortie ou sinks non supportés".to_string(),
            ));
        }
        let hidden = dims.hidden;
        let q_dim = dims.q_heads * dims.head_dim;
        let kv_dim = dims.kv_heads * dims.head_dim;
        let qkv_dim = q_dim + 2 * kv_dim;
        let inter = executor.linear_weight_out_dim(weights.gate_proj);
        let score_cells = dims.q_heads * kv.capacity();

        let normed = self.scratch().lease(hidden, GpuElement::F32)?;
        let qkv = self.scratch().lease(qkv_dim, GpuElement::F32)?;
        let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
        let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
        let ctx = self.scratch().lease(q_dim, GpuElement::F32)?;
        let o_out = self.scratch().lease(hidden, GpuElement::F32)?;
        let summed = self.scratch().lease(hidden, GpuElement::F32)?;
        let post_normed = self.scratch().lease(hidden, GpuElement::F32)?;
        let swiglu = self.scratch().lease(inter, GpuElement::F32)?;

        executor.encode_rms_norm_rows(
            encoder,
            layer_in,
            weights.input_norm,
            normed.tensor().buffer(),
            1,
            hidden,
            dims.eps,
        )?;
        let projected = executor.encode_matmul_weight_buffers(
            encoder,
            normed.tensor().buffer(),
            1,
            hidden,
            weights.qkv_proj,
            qkv.tensor().buffer(),
            false,
        )?;
        if projected != qkv_dim {
            return Err(InferError::Dimension(format!(
                "couche TTS fusionnée: qkv sort {projected}, attendu {qkv_dim}"
            )));
        }
        if !kv.encode_rope_qk_append_kv(
            encoder,
            qkv.tensor().buffer(),
            weights.q_norm,
            q_roped.tensor().buffer(),
            byte_offset_f32(q_dim, "couche TTS k offset")?,
            weights.k_norm,
            byte_offset_f32(q_dim + kv_dim, "couche TTS v offset")?,
            &dims,
            weights.rope_frequencies,
        )? {
            return Err(InferError::Config(
                "couche TTS fusionnée: KV quantifié non supporté".to_string(),
            ));
        }
        Self::encode_query_attention_scale(
            executor,
            encoder,
            owned,
            q_roped.tensor().buffer(),
            q_dim,
            dims.head_dim,
            dims.attn_scalar,
        )?;
        kv.encode_attention_decode_windowed(
            encoder,
            q_roped.tensor().buffer(),
            scores.tensor().buffer(),
            ctx.tensor().buffer(),
            dims.window_start,
        )?;
        executor.encode_matmul_weight_buffers(
            encoder,
            ctx.tensor().buffer(),
            1,
            q_dim,
            weights.o_proj,
            o_out.tensor().buffer(),
            false,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            o_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            1,
            hidden,
            dims.eps,
        )?;
        let fused_mlp = executor.encode_gate_up_swiglu_u6_buffers(
            encoder,
            post_normed.tensor().buffer(),
            weights.gate_proj,
            weights.up_proj,
            swiglu.tensor().buffer(),
            hidden,
        )? && executor.encode_qmv_u6_residual_buffers(
            encoder,
            swiglu.tensor().buffer(),
            weights.down_proj,
            summed.tensor().buffer(),
            layer_out,
            inter,
        )?;
        if !fused_mlp {
            return Err(InferError::Config(
                "couche TTS fusionnée: MLP non u6 gs64 aligné".to_string(),
            ));
        }
        Ok(())
    }
}
