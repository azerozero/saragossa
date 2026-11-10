//! Stockage commun des projections linear-attention, sans changer leurs noyaux.

use super::*;

impl MetalExecutor {
    /// À appeler au rattachement du runtime, avant de résoudre les poids.
    /// Seuls les buffers autonomes u4 gs64 sont déplacés : une vue de shard
    /// (notamment un futur mmap) ne doit pas devenir une copie de fichier.
    pub(crate) fn share_linear_attn_projections(
        &self,
        mut linears: [&mut Linear; 4],
    ) -> Result<()> {
        let mut sources = Vec::with_capacity(4);
        let mut lengths = Vec::with_capacity(4);
        let mut totals = [0u64; 3];
        let mut input_dim = None;
        for linear in &linears {
            let Some(w) = linear.affine_quantized() else {
                return Ok(());
            };
            if w.relocated_u4 || w.bits() != 4 || w.group_size() != 64 {
                return Ok(());
            }
            let [rows, cols] = w.shape() else {
                return Ok(());
            };
            if input_dim.is_some_and(|dim| dim != *cols) || cols % 64 != 0 {
                return Ok(());
            }
            input_dim = Some(*cols);
            let (Some(packed), Some(scales), Some(biases)) = (
                w.packed_metal_view(),
                w.scales_metal_view(),
                w.biases_metal_view(),
            ) else {
                return Ok(());
            };
            let bytes = [
                byte_len::<u32>(w.packed_len())?,
                byte_len::<u16>(checked_len(*rows, cols / 64, "LA affine")?)?,
                byte_len::<u16>(checked_len(*rows, cols / 64, "LA affine")?)?,
            ];
            for (index, (buffer, offset)) in [packed, scales, biases].iter().enumerate() {
                // 256 est plus strict que l'alignement scalaire (u32/bf16).
                // Pas de padding : toutes les lignes concaténées restent contiguës.
                if *offset != 0 || buffer.length() != bytes[index] || bytes[index] % 256 != 0 {
                    return Ok(());
                }
                totals[index] = totals[index].checked_add(bytes[index]).ok_or_else(|| {
                    InferError::Dimension("stockage LA partagé trop grand".into())
                })?;
            }
            sources.push([packed.0.clone(), scales.0.clone(), biases.0.clone()]);
            lengths.push(bytes);
        }
        let buffers = totals.map(|len| {
            self.device
                .new_buffer(len, MTLResourceOptions::StorageModeShared)
        });
        // Le pool libère aussi les références retenues par le blit à la fin
        // de cette couche, sans attendre le prochain pool de l'application.
        metal::objc::rc::autoreleasepool(|| -> Result<()> {
            let command = self.queue.new_command_buffer();
            let blit = command.new_blit_command_encoder();
            let mut offsets = [0; 3];
            for (source, bytes) in sources.iter().zip(&lengths) {
                for index in 0..3 {
                    blit.copy_from_buffer(
                        &source[index],
                        0,
                        &buffers[index],
                        offsets[index],
                        bytes[index],
                    );
                    offsets[index] += bytes[index];
                }
            }
            blit.end_encoding();
            commit_and_wait(command)
        })?;
        // Construire toutes les vues avant de publier : aucune mutation partielle
        // des poids si une validation de plage échoue. Les IDs restent stables.
        let mut replacements = Vec::with_capacity(4);
        let mut offsets = [0; 3];
        for (linear, bytes) in linears.iter().zip(&lengths) {
            let mut weight = linear
                .affine_quantized()
                .ok_or_else(|| InferError::Dimension("projection LA affine attendue".into()))?
                .clone();
            weight.set_packed_metal_view(buffers[0].clone(), (offsets[0] / 4) as usize)?;
            weight.set_affine_metal_views(
                buffers[1].clone(),
                (offsets[1] / 2) as usize,
                buffers[2].clone(),
                (offsets[2] / 2) as usize,
            )?;
            weight.relocated_u4 = true;
            replacements.push(weight);
            for index in 0..3 {
                offsets[index] += bytes[index];
            }
        }
        for (linear, replacement) in linears.iter_mut().zip(replacements) {
            let weight = linear
                .affine_quantized_mut()
                .ok_or_else(|| InferError::Dimension("projection LA affine attendue".into()))?;
            *weight = replacement;
        }
        Ok(())
    }
}

/// Les sous-concats sont des plages de lignes des mêmes trois buffers.
/// Ne touche pas aux paires split : leur choix de noyaux doit rester identique.
pub(super) fn shared_projection_concat(
    weights: &[&LinearWeight],
) -> Option<MetalLinearWeightBuffers> {
    let first = weights.first()?.affine_quantized()?;
    if !first.relocated_u4 {
        return None;
    }
    let (packed, packed_offset) = first.packed_metal_view()?;
    let (scales, scales_offset) = first.scales_metal_view()?;
    let (biases, biases_offset) = first.biases_metal_view()?;
    let [_, in_dim] = first.shape() else {
        return None;
    };
    let groups = in_dim / 64;
    let mut rows = 0usize;
    for source in weights {
        let w = source.affine_quantized()?;
        let [out, cols] = w.shape() else {
            return None;
        };
        if !w.relocated_u4 || cols != in_dim {
            return None;
        }
        for (view, buffer, start, stride) in [
            (w.packed_metal_view()?, packed, packed_offset, in_dim / 2),
            (w.scales_metal_view()?, scales, scales_offset, groups * 2),
            (w.biases_metal_view()?, biases, biases_offset, groups * 2),
        ] {
            if view.0.as_ptr() != buffer.as_ptr()
                || view.1 != start.checked_add(rows.checked_mul(stride)?)?
            {
                return None;
            }
        }
        rows = rows.checked_add(*out)?;
    }
    Some(MetalLinearWeightBuffers::AffineQuantized {
        relocated_u4: true,
        packed: packed.clone(),
        packed_offset: packed_offset as u64,
        scales: scales.clone(),
        scales_offset: scales_offset as u64,
        biases: biases.clone(),
        biases_offset: biases_offset as u64,
        out_dim: rows,
        in_dim: *in_dim,
        packed_cols: in_dim / 8,
        group_size: 64,
        bits: 4,
        groups,
    })
}
