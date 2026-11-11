//! Transport disque des états récurrents, sans pointeur ni réserve GPU.
use super::*;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct LinearSnapshotLayout {
    conv_len: usize,
    conv_dim: usize,
    conv_kernel_dim: usize,
    num_value_heads: usize,
    value_head_dim: usize,
    key_head_dim: usize,
    pub(crate) ssm_bf16: bool,
}

impl LinearSnapshotLayout {
    pub(crate) fn matches_config(
        &self,
        config: crate::linear_attention::LinearAttentionConfig,
    ) -> bool {
        let conv = config
            .num_key_heads
            .checked_mul(config.key_head_dim)
            .and_then(|n| n.checked_mul(2))
            .and_then(|n| {
                config
                    .num_value_heads
                    .checked_mul(config.value_head_dim)
                    .and_then(|v| n.checked_add(v))
            });
        self.conv_kernel_dim == config.conv_kernel_dim
            && self.num_value_heads == config.num_value_heads
            && self.value_head_dim == config.value_head_dim
            && self.key_head_dim == config.key_head_dim
            && conv == Some(self.conv_dim)
    }
}

impl LinearAttentionMetalState {
    /// Le blit synchronisé isole l'état avant toute lecture CPU des buffers.
    pub(crate) fn disk_parts(
        &self,
        metal: &MetalExecutor,
    ) -> Result<(LinearSnapshotLayout, Vec<f32>, Vec<u8>)> {
        let state = metal.snapshot_linear_attn_state(self)?;
        let conv = read_f32_buffer(&state.conv, state.conv_len)?;
        let ssm = if state.ssm_bf16 {
            bytemuck::cast_slice(&read_u16_buffer(&state.ssm, state.ssm_len)?).to_vec()
        } else {
            bytemuck::cast_slice(&read_f32_buffer(&state.ssm, state.ssm_len)?).to_vec()
        };
        Ok((
            LinearSnapshotLayout {
                conv_len: state.conv_len,
                conv_dim: state.conv_dim,
                conv_kernel_dim: state.conv_kernel_dim,
                num_value_heads: state.num_value_heads,
                value_head_dim: state.value_head_dim,
                key_head_dim: state.key_head_dim,
                ssm_bf16: state.ssm_bf16,
            },
            conv,
            ssm,
        ))
    }
}

impl MetalExecutor {
    pub(crate) fn linear_state_from_disk(
        &self,
        layout: &LinearSnapshotLayout,
        conv: &[f32],
        ssm: &[u8],
    ) -> Result<LinearAttentionMetalState> {
        let invalid =
            || InferError::Dimension("instantané linear-attn : dimensions invalides".into());
        // Les chemins par opérations gardent K-1 lignes, les chemins fusionnés
        // gardent K lignes : conserver cette géométrie évite un faux rejet.
        let short = layout
            .conv_kernel_dim
            .checked_sub(1)
            .and_then(|n| n.checked_mul(layout.conv_dim))
            .ok_or_else(invalid)?;
        let full = layout
            .conv_kernel_dim
            .checked_mul(layout.conv_dim)
            .ok_or_else(invalid)?;
        let conv_len = layout.conv_len;
        if conv_len != short && conv_len != full {
            return Err(invalid());
        }
        let ssm_len = layout
            .num_value_heads
            .checked_mul(layout.value_head_dim)
            .and_then(|n| n.checked_mul(layout.key_head_dim))
            .ok_or_else(invalid)?;
        let ssm_bytes = ssm_len
            .checked_mul(if layout.ssm_bf16 { 2 } else { 4 })
            .ok_or_else(invalid)?;
        if conv_len != conv.len() || ssm_bytes != ssm.len() || ssm_len == 0 {
            return Err(invalid());
        }
        Ok(LinearAttentionMetalState {
            conv: self.buffer_from_f32(conv, "snapshot_conv")?,
            ssm: self.buffer_from_slice(ssm, "snapshot_ssm")?,
            conv_len,
            ssm_len,
            conv_dim: layout.conv_dim,
            conv_kernel_dim: layout.conv_kernel_dim,
            num_value_heads: layout.num_value_heads,
            value_head_dim: layout.value_head_dim,
            key_head_dim: layout.key_head_dim,
            ssm_bf16: layout.ssm_bf16,
            ssm_journal: false,
            tree: false,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_disk_roundtrip_preserves_layout_and_bits() -> Result<()> {
        let metal = MetalExecutor::new()?;
        for bf16 in [false, true] {
            for conv_len in [12, 16] {
                let layout = LinearSnapshotLayout {
                    conv_len,
                    conv_dim: 4,
                    conv_kernel_dim: 4,
                    num_value_heads: 1,
                    value_head_dim: 2,
                    key_head_dim: 2,
                    ssm_bf16: bf16,
                };
                let conv: Vec<f32> = (0..conv_len).map(|n| n as f32 - 4.0).collect();
                let bytes = if bf16 {
                    bytemuck::cast_slice(&[0x8000u16, 0x3f80, 0xbf80, 0x3f81]).to_vec()
                } else {
                    bytemuck::cast_slice(&[-0.0f32, 1.0, -1.0, 1.000_001]).to_vec()
                };
                let state = metal.linear_state_from_disk(&layout, &conv, &bytes)?;
                let (restored_layout, restored_conv, restored_bytes) = state.disk_parts(&metal)?;
                assert_eq!(restored_layout.conv_len, conv_len);
                assert_eq!(restored_layout.ssm_bf16, bf16);
                assert_eq!(restored_conv, conv);
                assert_eq!(restored_bytes, bytes);
            }
        }
        Ok(())
    }
}
