//! Contrôle intégral des tailles avant la première lecture/allocation de blob.
use super::*;

fn product(values: &[usize]) -> Result<usize> {
    values.iter().try_fold(1usize, |n, &v| {
        n.checked_mul(v)
            .ok_or_else(|| invalid("dimensions trop grandes"))
    })
}

#[derive(Default)]
struct Budget {
    maximum: usize,
    used: usize,
}
impl Budget {
    fn allow(&mut self, elements: usize) -> Result<()> {
        self.maximum = self
            .maximum
            .checked_add(product(&[elements, 4])?)
            .ok_or_else(|| invalid("budget cumulé trop grand"))?;
        Ok(())
    }
    fn blob(&mut self, blob: &Blob, dtype: &str, elements: usize) -> Result<()> {
        blob.validate(dtype, elements)?;
        self.used = self
            .used
            .checked_add(blob.bytes)
            .ok_or_else(|| invalid("taille cumulée trop grande"))?;
        if self.used > self.maximum {
            return Err(invalid("budget cumulé dépassé"));
        }
        Ok(())
    }
}

impl Rows {
    fn validate(&self, tokens: usize, dim: usize, budget: &mut Budget) -> Result<()> {
        let expected = product(&[tokens, dim])?;
        if dim == 0 || self.len != expected || self.base > expected || self.base % dim != 0 {
            return Err(invalid("géométrie KV différente du modèle/prompt"));
        }
        budget.allow(expected)?;
        let mut remaining = expected - self.base;
        for blob in &self.segments {
            let width = match blob.dtype.as_str() {
                "f32" => 4,
                "bf16" => 2,
                _ => return Err(invalid("dtype KV inconnu")),
            };
            let row_bytes = product(&[dim, width])?;
            if blob.bytes == 0 || blob.bytes % row_bytes != 0 {
                return Err(invalid("segment KV sans lignes entières"));
            }
            let elements = blob.bytes / width;
            remaining = remaining
                .checked_sub(elements)
                .ok_or_else(|| invalid("segment KV dépasse les dimensions attendues"))?;
            budget.blob(blob, &blob.dtype, elements)?;
        }
        if remaining != 0 {
            return Err(invalid("segments KV incomplets"));
        }
        Ok(())
    }
}

impl Dense {
    fn validate(&self, shape: &[usize], budget: &mut Budget) -> Result<()> {
        if self.shape != shape {
            return Err(invalid("forme du tenseur différente du modèle/prompt"));
        }
        let elements = product(shape)?;
        budget.allow(elements)?;
        budget.blob(&self.data, "f32", elements)
    }
}

impl Layer {
    fn validate_full(&self, tokens: usize, dim: usize, budget: &mut Budget) -> Result<()> {
        if self.kv_dim != Some(dim) && !(tokens == 0 && self.kv_dim.is_none()) {
            return Err(invalid("largeur KV différente du modèle"));
        }
        self.keys.validate(tokens, dim, budget)?;
        self.values.validate(tokens, dim, budget)?;
        if self.keys.base != self.values.base || self.conv_dim.is_some() || self.ssm_shape.is_some()
        {
            return Err(invalid("état full-attention incohérent"));
        }
        budget.blob(&self.conv, "f32", 0)?;
        budget.blob(&self.ssm, "f32", 0)?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if self.metal.is_some() {
            return Err(invalid("état linéaire sur full-attention"));
        }
        Ok(())
    }

    fn validate_linear(&self, config: LinearAttentionConfig, budget: &mut Budget) -> Result<()> {
        self.keys.validate(0, 1, budget)?;
        self.values.validate(0, 1, budget)?;
        if self.kv_dim.is_some() {
            return Err(invalid("KV sur couche linéaire"));
        }
        let conv_dim = product(&[config.num_key_heads, config.key_head_dim, 2])?
            .checked_add(product(&[config.num_value_heads, config.value_head_dim])?)
            .ok_or_else(|| invalid("dimension conv trop grande"))?;
        let short = product(&[
            config
                .conv_kernel_dim
                .checked_sub(1)
                .ok_or_else(|| invalid("noyau conv vide"))?,
            conv_dim,
        ])?;
        let ssm_shape = (
            config.num_value_heads,
            config.value_head_dim,
            config.key_head_dim,
        );
        let ssm = product(&[ssm_shape.0, ssm_shape.1, ssm_shape.2])?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        let has_gpu = if let Some(gpu) = &self.metal {
            let full = product(&[config.conv_kernel_dim, conv_dim])?;
            if !gpu.layout.matches_config(config)
                || (gpu.layout.conv_len != short && gpu.layout.conv_len != full)
            {
                return Err(invalid("géométrie linéaire Metal différente du modèle"));
            }
            budget.allow(full)?;
            budget.allow(ssm)?;
            budget.blob(&gpu.conv, "f32", gpu.layout.conv_len)?;
            budget.blob(
                &gpu.ssm,
                if gpu.layout.ssm_bf16 { "bf16" } else { "f32" },
                ssm,
            )?;
            true
        } else {
            false
        };
        #[cfg(not(all(target_os = "macos", feature = "metal")))]
        let has_gpu = false;
        budget.allow(short)?;
        budget.allow(ssm)?;
        if has_gpu && self.conv.bytes == 0 && self.ssm.bytes == 0 {
            budget.blob(&self.conv, "f32", 0)?;
            budget.blob(&self.ssm, "f32", 0)?;
        } else {
            if self.conv_dim != Some(conv_dim) || self.ssm_shape != Some(ssm_shape) {
                return Err(invalid("géométrie récurrente CPU incohérente"));
            }
            budget.blob(&self.conv, "f32", short)?;
            budget.blob(&self.ssm, "f32", ssm)?;
        }
        Ok(())
    }
}

impl Manifest {
    pub(super) fn validate(&self, model: &CausalDecoder, tokens: usize) -> Result<()> {
        let mut budget = Budget::default();
        for (layer, definition) in self.layers.iter().zip(&model.layers) {
            match &definition.attention {
                AttentionBlock::Full(attention) => {
                    let dim = attention
                        .k_proj
                        .weight()
                        .shape()
                        .first()
                        .copied()
                        .ok_or_else(|| invalid("projection K sans dimension"))?;
                    layer.validate_full(tokens, dim, &mut budget)?;
                }
                AttentionBlock::Linear(_) => {
                    layer.validate_linear(model.config.linear_attention_config()?, &mut budget)?
                }
            }
        }
        let hidden = model.final_norm.data().len();
        self.final_state.validate(&[1, hidden], &mut budget)?;
        if let Some(history) = &self.history {
            let head = model
                .mtp
                .as_ref()
                .ok_or_else(|| invalid("historique sans tête MTP"))?;
            let dim = head
                .layer
                .attention
                .k_proj
                .weight()
                .shape()
                .first()
                .copied()
                .ok_or_else(|| invalid("projection MTP sans dimension"))?;
            history.validate_full(tokens.saturating_sub(1), dim, &mut budget)?;
        }
        if let Some(capture) = &self.captures {
            let rows = tokens
                .checked_sub(capture.from)
                .ok_or_else(|| invalid("base des captures hors prompt"))?;
            if capture.layer_ids.is_empty()
                || capture.layer_ids.len() > model.layers.len()
                || capture.layer_ids.iter().enumerate().any(|(i, &id)| {
                    id >= model.layers.len() || capture.layer_ids[..i].contains(&id)
                })
            {
                return Err(invalid("couches des captures invalides"));
            }
            let cols = product(&[capture.layer_ids.len(), hidden])?;
            capture.hidden.validate(&[rows, cols], &mut budget)?;
        }
        Ok(())
    }
}
