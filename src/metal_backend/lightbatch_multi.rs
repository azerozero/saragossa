//! Encodage multi-flux dense (light-batch à N flux) : projections linear-attn
//! batchées sur N lignes (poids lus une fois pour les N flux), cœur conv/SSM
//! PAR FLUX sur l'état de chaque flux.
//!
//! Chaque ligne passe par le même cœur que le solo
//! ([`MetalExecutor::encode_linear_attn_row_core`]) ; seules les projections
//! changent de kernel (rows=N au lieu du qmv mono-ligne), d'où une dérive f32
//! de l'ordre de 1e-6 mesurée par l'oracle multi-flux.

use super::linear_attention::{LinearAttnCoreWeights, LinearAttnRowSources};
use super::*;

/// Vue en lignes d'une sortie de projection : la ligne `row` commence à
/// l'élément `row * stride + base`.
struct RowView<'a> {
    buffer: &'a BufferRef,
    stride: usize,
    base: usize,
}

impl<'a> RowView<'a> {
    fn at(&self, row: usize) -> Result<(&'a BufferRef, u64)> {
        let element = row
            .checked_mul(self.stride)
            .and_then(|value| value.checked_add(self.base))
            .ok_or_else(|| InferError::Dimension("multi linear-attn ligne déborde".to_string()))?;
        Ok((
            self.buffer,
            byte_offset_f32(element, "multi linear-attn ligne")?,
        ))
    }
}

impl MetalExecutor {
    /// Projette `[rows, in_dim]` par `weight` vers un buffer scratch étiqueté et
    /// vérifie la largeur de sortie.
    #[expect(
        clippy::too_many_arguments,
        reason = "projection en lignes : entrée + dims + poids + label du scratch"
    )]
    fn project_multi_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        rows: usize,
        in_dim: usize,
        weight: &MetalLinearWeightBuffers,
        out_dim: usize,
        label: &'static str,
    ) -> Result<Buffer> {
        let output = self.private_f32_buffer(checked_len(rows, out_dim, label)?, label)?;
        let projected = self
            .encode_matmul_weight_buffers(encoder, input, rows, in_dim, weight, &output, false)?;
        if projected != out_dim {
            return Err(InferError::Dimension(format!(
                "{label} sort {projected}, attendu {out_dim}"
            )));
        }
        Ok(output)
    }

    /// Encode UNE couche linear-attn dense pour `states.len()` flux : norme
    /// d'entrée en miroir du solo, projections d'entrée batchées, cœur conv/SSM
    /// par flux (scratch namespacé par `slots`), out_proj batchée.
    /// `input_buffer`/`output_buffer` = `[N, in_dim]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si moins de deux flux sont fournis, si une dimension
    /// est incohérente ou si un encodage Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow multi-flux : N états + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_resident_multi_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_norm: (&BufferRef, f32),
        output_buffer: &BufferRef,
        weights: &MetalLinearAttnResidentDenseWeights,
        states: &[&LinearAttentionMetalState],
        slots: &[u64],
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        let rows = states.len();
        if rows < 2 || slots.len() != rows {
            return Err(InferError::Dimension(format!(
                "multi linear-attn: {rows} états pour {} slots (au moins 2)",
                slots.len()
            )));
        }
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            ..
        } = dims;
        let heads = spec.num_value_heads;
        let (norm_weight, eps) = input_norm;
        let normed = self.private_f32_buffer(
            checked_len(rows, in_dim, "multi linear-attn normed")?,
            "lightbatch_multi_la_normed",
        )?;
        let gated = self.private_f32_buffer(
            checked_len(rows, value_dim, "multi linear-attn gated")?,
            "lightbatch_multi_la_gated",
        )?;
        // Les buffers de projection vivent jusqu'à la fin de l'encodage des
        // lignes ; les vues ne font que les référencer.
        let mut projections: Vec<Buffer> = Vec::with_capacity(2);
        // Même aiguillage que le solo `encode_linear_attn_resident_dense_buffers` :
        // concat complète, puis z+beta+gate, puis paires.
        let full = weights
            .full
            .as_ref()
            .filter(|_| linear_full_concat_enabled());
        let z_beta_gate = match (&weights.qkv_z, weights.z_beta_gate.as_ref()) {
            (MetalLinearAttnResidentPairWeights::Split { first, .. }, Some(z_beta_gate))
                if linear_z_beta_gate_enabled() =>
            {
                Some((first, z_beta_gate))
            }
            _ => None,
        };
        if let Some(full) = full {
            // Le solo fusionne le prologue rms dans l'in_proj quand le poids s'y
            // prête : la norme rows reprend alors la même réduction.
            self.encode_rms_norm_rows_matching_solo(
                encoder,
                input_buffer,
                norm_weight,
                &normed,
                rows,
                in_dim,
                eps,
                &full.in_proj,
                false,
            )?;
        } else {
            self.encode_rms_norm_rows(
                encoder,
                input_buffer,
                norm_weight,
                &normed,
                rows,
                in_dim,
                eps,
            )?;
        }
        let out_proj = full.map_or(&weights.out_proj, |full| &full.out_proj);
        let core_weights = full.map_or(LinearAttnCoreWeights::from_dense(weights), |full| {
            LinearAttnCoreWeights::from_full(full)
        });
        let qkv_z_len = conv_dim
            .checked_add(value_dim)
            .ok_or_else(|| InferError::Dimension("multi qkv+z déborde".to_string()))?;
        let z_beta_gate_len = value_dim
            .checked_add(2 * heads)
            .ok_or_else(|| InferError::Dimension("multi z+beta+gate déborde".to_string()))?;
        // (index de projection, stride, base) de qkv, z, beta, gate.
        let layout: [(usize, usize, usize); 4] = if let Some(full) = full {
            let in_proj_len = qkv_z_len
                .checked_add(2 * heads)
                .ok_or_else(|| InferError::Dimension("multi in_proj déborde".to_string()))?;
            projections.push(self.project_multi_rows(
                encoder,
                &normed,
                rows,
                in_dim,
                &full.in_proj,
                in_proj_len,
                "lightbatch_multi_la_in_proj",
            )?);
            [
                (0, in_proj_len, 0),
                (0, in_proj_len, conv_dim),
                (0, in_proj_len, qkv_z_len),
                (0, in_proj_len, qkv_z_len + heads),
            ]
        } else if let Some((qkv_weight, z_beta_gate_weight)) = z_beta_gate {
            projections.push(self.project_multi_rows(
                encoder,
                &normed,
                rows,
                in_dim,
                qkv_weight,
                conv_dim,
                "lightbatch_multi_la_qkv",
            )?);
            projections.push(self.project_multi_rows(
                encoder,
                &normed,
                rows,
                in_dim,
                z_beta_gate_weight,
                z_beta_gate_len,
                "lightbatch_multi_la_z_beta_gate",
            )?);
            [
                (0, conv_dim, 0),
                (1, z_beta_gate_len, 0),
                (1, z_beta_gate_len, value_dim),
                (1, z_beta_gate_len, value_dim + heads),
            ]
        } else {
            let [qkv, z] = match &weights.qkv_z {
                MetalLinearAttnResidentPairWeights::Concat(weight) => {
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        weight,
                        qkv_z_len,
                        "lightbatch_multi_la_qkv_z",
                    )?);
                    let index = projections.len() - 1;
                    [(index, qkv_z_len, 0), (index, qkv_z_len, conv_dim)]
                }
                MetalLinearAttnResidentPairWeights::Split { first, second } => {
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        first,
                        conv_dim,
                        "lightbatch_multi_la_qkv",
                    )?);
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        second,
                        value_dim,
                        "lightbatch_multi_la_z",
                    )?);
                    let index = projections.len() - 2;
                    [(index, conv_dim, 0), (index + 1, value_dim, 0)]
                }
            };
            let [beta, gate] = match &weights.beta_gate {
                MetalLinearAttnResidentPairWeights::Concat(weight) => {
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        weight,
                        2 * heads,
                        "lightbatch_multi_la_beta_gate",
                    )?);
                    let index = projections.len() - 1;
                    [(index, 2 * heads, 0), (index, 2 * heads, heads)]
                }
                MetalLinearAttnResidentPairWeights::Split { first, second } => {
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        first,
                        heads,
                        "lightbatch_multi_la_beta",
                    )?);
                    projections.push(self.project_multi_rows(
                        encoder,
                        &normed,
                        rows,
                        in_dim,
                        second,
                        heads,
                        "lightbatch_multi_la_gate",
                    )?);
                    let index = projections.len() - 2;
                    [(index, heads, 0), (index + 1, heads, 0)]
                }
            };
            [qkv, z, beta, gate]
        };
        let view = |(index, stride, base): (usize, usize, usize)| -> Result<RowView<'_>> {
            let buffer = projections.get(index).ok_or_else(|| {
                InferError::Dimension("multi linear-attn projection absente".to_string())
            })?;
            Ok(RowView {
                buffer,
                stride,
                base,
            })
        };
        let [qkv, z, beta, gate] = [
            view(layout[0])?,
            view(layout[1])?,
            view(layout[2])?,
            view(layout[3])?,
        ];
        let gated_rows = RowView {
            buffer: &gated,
            stride: value_dim,
            base: 0,
        };
        for (row, (state, slot)) in states.iter().zip(slots).enumerate() {
            // Mêmes labels que le solo : le namespace du flux rend les scratch
            // du cœur disjoints entre flux d'un même command buffer.
            let _namespace = install_scratch_namespace(*slot);
            self.encode_linear_attn_row_core(
                encoder,
                LinearAttnRowSources {
                    qkv: qkv.at(row)?,
                    z: z.at(row)?,
                    beta: beta.at(row)?,
                    gate: gate.at(row)?,
                },
                core_weights,
                state,
                spec,
                dims,
                gated_rows.at(row)?,
                full.is_some(),
            )?;
        }
        let out_dim = self.encode_matmul_weight_buffers(
            encoder,
            &gated,
            rows,
            value_dim,
            out_proj,
            output_buffer,
            false,
        )?;
        if out_dim != in_dim {
            return Err(InferError::Dimension(format!(
                "multi out_proj sort {out_dim}, attendu {in_dim}"
            )));
        }
        Ok(())
    }
}
