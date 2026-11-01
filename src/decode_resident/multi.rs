//! Couches denses multi-flux (light-batch à N flux) : un pas encode N tokens de
//! N séquences différentes dans le même encoder.
//!
//! Les sections sans état (normes, projections q/k/v/o, tail dense) sont
//! batchées sur N lignes ; RoPE, append K/V et attention suivent la position et
//! le KV PROPRES à chaque flux, avec les kernels du decode solo. La ligne `r`
//! du batch est un rang compact : l'identité du flux est portée par son KV et
//! son slot de scratch.

use super::utils::byte_offset_f32;
use super::*;

/// État d'UN flux pour une couche full-attn multi-flux.
pub(crate) struct MultiFullAttnStream<'a> {
    pub kv: &'a mut FullAttentionMetalState,
    /// Dimensions de la couche à la position DU flux.
    pub dims: FullAttnLayerDims,
    /// Namespace scratch du flux (slot de son arène).
    pub slot: u64,
}

/// Décalage en octets de la ligne `row` d'un buffer f32 `[rows, stride]`, plus
/// `base` éléments.
fn row_offset(row: usize, stride: usize, base: usize, label: &'static str) -> Result<u64> {
    let element = row
        .checked_mul(stride)
        .and_then(|value| value.checked_add(base))
        .ok_or_else(|| InferError::Dimension(format!("{label} déborde")))?;
    byte_offset_f32(element, label)
}

impl DecodeResidentState {
    /// Encode UNE couche full-attn DENSE pour `streams.len()` flux :
    /// `layer_in [N, hidden]` → `layer_out [N, hidden]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si moins de deux flux sont fournis, si la couche sort
    /// du périmètre multi-flux (fenêtre glissante, sinks, Gemma, MoE flint), si
    /// une dimension est incohérente ou si un encodage Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow multi-flux d'une couche : exécuteur + N KV + poids + ping-pong"
    )]
    pub(crate) fn encode_full_attn_dense_layer_multi(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        streams: &mut [MultiFullAttnStream<'_>],
        weights: FullAttnDenseLayerWeights<'_>,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let rows = streams.len();
        let dims = streams
            .first()
            .map(|stream| stream.dims)
            .filter(|_| rows >= 2)
            .ok_or_else(|| {
                InferError::Dimension(format!("multi full-attn: {rows} flux (au moins 2)"))
            })?;
        if !dims.attn_output_gate
            || weights.sigmoid_moe.is_some()
            || weights.parallel_moe.is_some()
            || weights.pre_feedforward_norm.is_some()
            || weights.post_feedforward_norm.is_some()
        {
            return Err(InferError::Config(
                "multi full-attn: couche hors périmètre (porte, Gemma ou MoE)".to_string(),
            ));
        }
        // Le KV d'un flux est lu en entier : une fenêtre ou des sinks
        // changeraient le masque que seul le solo sait appliquer ici.
        if streams
            .iter()
            .any(|stream| stream.dims.window_start != 0 || stream.kv.has_sinks())
        {
            return Err(InferError::Config(
                "multi full-attn: fenêtre glissante ou sinks non supportés".to_string(),
            ));
        }
        let hidden = dims.hidden;
        let q_dim = dims
            .q_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("multi q_dim déborde".to_string()))?;
        let kv_dim = dims
            .kv_heads
            .checked_mul(dims.head_dim)
            .ok_or_else(|| InferError::Dimension("multi kv_dim déborde".to_string()))?;
        let q_proj_dim = q_dim
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("multi q_gate_dim déborde".to_string()))?;
        let qkv_proj_dim = q_proj_dim
            .checked_add(kv_dim)
            .and_then(|value| value.checked_add(kv_dim))
            .ok_or_else(|| InferError::Dimension("multi qkv concat déborde".to_string()))?;
        let elements = |width: usize| {
            rows.checked_mul(width).ok_or_else(|| {
                InferError::Dimension("multi full-attn lignes débordent".to_string())
            })
        };

        let normed = self.scratch().lease(elements(hidden)?, GpuElement::F32)?;
        let q_raw = self.scratch().lease(elements(q_dim)?, GpuElement::F32)?;
        let gate = self.scratch().lease(elements(q_dim)?, GpuElement::F32)?;
        let ctx = self.scratch().lease(elements(q_dim)?, GpuElement::F32)?;
        let gated_ctx = self.scratch().lease(elements(q_dim)?, GpuElement::F32)?;
        let o_out = self.scratch().lease(elements(hidden)?, GpuElement::F32)?;
        let summed = self.scratch().lease(elements(hidden)?, GpuElement::F32)?;
        let post_normed = self.scratch().lease(elements(hidden)?, GpuElement::F32)?;

        let project = |input: &BufferRef, weight: &MetalLinearWeightBuffers, out_dim: usize| {
            let output = self.scratch().lease(elements(out_dim)?, GpuElement::F32)?;
            let projected = executor.encode_matmul_weight_buffers(
                encoder,
                input,
                rows,
                hidden,
                weight,
                output.tensor().buffer(),
                false,
            )?;
            if projected != out_dim {
                return Err(InferError::Dimension(format!(
                    "multi projection sort {projected}, attendu {out_dim}"
                )));
            }
            Ok(output)
        };
        // Sources en lignes : (index de projection, stride, base) de q+gate, K, V.
        let mut projections: Vec<ScratchLease> = Vec::with_capacity(3);
        let (q_source, k_source, v_source) = if let Some(qkv_proj) = weights.qkv_proj {
            // Le solo fusionne la norme dans la projection qkv quand il le peut :
            // la norme en lignes reprend alors la même réduction.
            executor.encode_rms_norm_rows_matching_solo(
                encoder,
                layer_in,
                weights.input_norm,
                normed.tensor().buffer(),
                rows,
                hidden,
                dims.eps,
                qkv_proj,
                true,
            )?;
            projections.push(project(normed.tensor().buffer(), qkv_proj, qkv_proj_dim)?);
            (
                (0, qkv_proj_dim, 0),
                (0, qkv_proj_dim, q_proj_dim),
                (0, qkv_proj_dim, q_proj_dim + kv_dim),
            )
        } else {
            executor.encode_rms_norm_rows(
                encoder,
                layer_in,
                weights.input_norm,
                normed.tensor().buffer(),
                rows,
                hidden,
                dims.eps,
            )?;
            let normed = normed.tensor().buffer();
            projections.push(project(normed, weights.q_proj, q_proj_dim)?);
            projections.push(project(normed, weights.k_proj, kv_dim)?);
            projections.push(project(normed, weights.v_proj, kv_dim)?);
            ((0, q_proj_dim, 0), (1, kv_dim, 0), (2, kv_dim, 0))
        };
        let source = |index: usize| {
            projections
                .get(index)
                .map(|lease| lease.tensor().buffer())
                .ok_or_else(|| InferError::Dimension("multi projection absente".to_string()))
        };
        let (q_buffer, k_buffer, v_buffer) = (
            source(q_source.0)?,
            source(k_source.0)?,
            source(v_source.0)?,
        );
        executor.encode_split_q_gate_rows_with_stride(
            encoder,
            q_buffer,
            q_raw.tensor().buffer(),
            gate.tensor().buffer(),
            rows,
            dims.q_heads,
            dims.head_dim,
            q_source.1,
        )?;

        // Les scratch par flux restent loués jusqu'à la fin de la couche : aucun
        // flux ne réutilise ceux d'un autre dans le même command buffer.
        let mut stream_leases: Vec<ScratchLease> = Vec::with_capacity(rows * 6);
        for (row, stream) in streams.iter_mut().enumerate() {
            let _namespace = crate::metal_backend::install_scratch_namespace(stream.slot);
            let dims = stream.dims;
            let kv = &mut *stream.kv;
            let score_cells = dims
                .q_heads
                .checked_mul(kv.capacity())
                .ok_or_else(|| InferError::Dimension("multi scores débordent".to_string()))?;
            let q_roped = self.scratch().lease(q_dim, GpuElement::F32)?;
            let k_roped = self.scratch().lease(kv_dim, GpuElement::F32)?;
            let scores = self.scratch().lease(score_cells, GpuElement::F32)?;
            let ctx_row = self.scratch().lease(q_dim, GpuElement::F32)?;

            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                q_raw.tensor().buffer(),
                row_offset(row, q_dim, 0, "multi q row")?,
                weights.q_norm,
                q_roped.tensor().buffer(),
                dims.q_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            Self::encode_query_attention_scale(
                executor,
                encoder,
                owned,
                q_roped.tensor().buffer(),
                q_dim,
                dims.head_dim,
                dims.attn_scalar,
            )?;
            self.encode_rms_norm_rope_decode_with_offset(
                encoder,
                k_buffer,
                row_offset(row, k_source.1, k_source.2, "multi k row")?,
                weights.k_norm,
                k_roped.tensor().buffer(),
                dims.kv_heads,
                dims.head_dim,
                dims.rope_dims,
                dims.rope_frequency_dim,
                dims.position,
                dims.eps,
                dims.theta,
            )?;
            let v_offset = row_offset(row, v_source.1, v_source.2, "multi v row")?;
            if weights.value_norm {
                let v_row = self.scratch().lease(kv_dim, GpuElement::F32)?;
                let v_normed = self.scratch().lease(kv_dim, GpuElement::F32)?;
                executor.encode_copy_with_offsets(
                    encoder,
                    v_buffer,
                    v_offset,
                    v_row.tensor().buffer(),
                    0,
                    kv_dim,
                )?;
                executor.encode_rms_norm_heads_no_scale_rows(
                    encoder,
                    v_row.tensor().buffer(),
                    v_normed.tensor().buffer(),
                    1,
                    dims.kv_heads,
                    dims.head_dim,
                    dims.eps,
                )?;
                kv.encode_append_kv_with_offsets(
                    encoder,
                    k_roped.tensor().buffer(),
                    0,
                    v_normed.tensor().buffer(),
                    0,
                )?;
                stream_leases.push(v_row);
                stream_leases.push(v_normed);
            } else {
                kv.encode_append_kv_with_offsets(
                    encoder,
                    k_roped.tensor().buffer(),
                    0,
                    v_buffer,
                    v_offset,
                )?;
            }
            kv.encode_attention_decode_windowed(
                encoder,
                q_roped.tensor().buffer(),
                scores.tensor().buffer(),
                ctx_row.tensor().buffer(),
                0,
            )?;
            executor.encode_copy_with_offsets(
                encoder,
                ctx_row.tensor().buffer(),
                0,
                ctx.tensor().buffer(),
                row_offset(row, q_dim, 0, "multi ctx row")?,
                q_dim,
            )?;
            stream_leases.extend([q_roped, k_roped, scores, ctx_row]);
        }

        executor.encode_attn_gate_rows(
            encoder,
            ctx.tensor().buffer(),
            gate.tensor().buffer(),
            gated_ctx.tensor().buffer(),
            elements(q_dim)?,
        )?;
        let o_dim = executor.encode_matmul_weight_buffers(
            encoder,
            gated_ctx.tensor().buffer(),
            rows,
            q_dim,
            weights.o_proj,
            o_out.tensor().buffer(),
            false,
        )?;
        if o_dim != hidden {
            return Err(InferError::Dimension(format!(
                "multi o_proj sort {o_dim}, attendu {hidden}"
            )));
        }
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            o_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            rows,
            hidden,
            dims.eps,
        )?;
        self.encode_dense_tail_rows(
            executor,
            encoder,
            owned,
            post_normed.tensor().buffer(),
            summed.tensor().buffer(),
            layer_out,
            rows,
            hidden,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            weights.tail_score,
        )
    }

    /// Encode UNE couche linear-attn DENSE pour `states.len()` flux : cœur
    /// conv/SSM par flux via
    /// [`MetalExecutor::encode_linear_attn_resident_multi_buffers`], tail dense
    /// batché.
    ///
    /// # Errors
    ///
    /// Propage toute erreur d'encodage (dimension, Metal).
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow multi-flux d'une couche : exécuteur + N états + poids + spec + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_dense_layer_multi(
        &self,
        executor: &MetalExecutor,
        encoder: &ComputeCommandEncoderRef,
        owned: &mut Vec<Buffer>,
        states: &[&LinearAttentionMetalState],
        slots: &[u64],
        weights: LinearAttnDenseLayerWeights<'_>,
        spec: LinearAttentionStepSpec,
        res_dims: LinearAttnResidentDims,
        hidden: usize,
        eps: f32,
        layer_in: &BufferRef,
        layer_out: &BufferRef,
    ) -> Result<()> {
        let rows = states.len();
        let elements = rows
            .checked_mul(hidden)
            .ok_or_else(|| InferError::Dimension("multi linear-attn hidden déborde".to_string()))?;
        let attn_out = self.scratch().lease(elements, GpuElement::F32)?;
        let summed = self.scratch().lease(elements, GpuElement::F32)?;
        let post_normed = self.scratch().lease(elements, GpuElement::F32)?;
        executor.encode_linear_attn_resident_multi_buffers(
            encoder,
            layer_in,
            (weights.input_norm, eps),
            attn_out.tensor().buffer(),
            weights.linear,
            states,
            slots,
            spec,
            res_dims,
        )?;
        executor.encode_add_rms_norm_rows(
            encoder,
            layer_in,
            attn_out.tensor().buffer(),
            weights.post_norm,
            summed.tensor().buffer(),
            post_normed.tensor().buffer(),
            rows,
            hidden,
            eps,
        )?;
        self.encode_dense_tail_rows(
            executor,
            encoder,
            owned,
            post_normed.tensor().buffer(),
            summed.tensor().buffer(),
            layer_out,
            rows,
            hidden,
            weights.gate_proj,
            weights.up_proj,
            weights.down_proj,
            weights.tail_score,
        )
    }
}
