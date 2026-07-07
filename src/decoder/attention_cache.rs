//! Mise à jour du cache K/V attention.

use super::attention_ops::AttentionLayout;
use super::*;

impl LayerKvCache {
    pub(super) fn append(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layout: &AttentionLayout,
    ) -> Result<()> {
        let (key_rows, key_dim) = key.as_matrix()?;
        let (value_rows, value_dim) = value.as_matrix()?;
        let expected_dim = layout.num_key_value_heads * layout.head_dim;
        if key_rows != 1 || value_rows != 1 || key_dim != expected_dim || value_dim != expected_dim
        {
            return Err(InferError::Dimension(format!(
                "cache KV attend key/value [1,{expected_dim}], reçu key={:?}, value={:?}",
                key.shape(),
                value.shape()
            )));
        }
        match self.kv_dim {
            Some(dim) if dim != expected_dim => {
                return Err(InferError::Dimension(format!(
                    "cache KV dim={dim} incompatible avec {expected_dim}"
                )))
            }
            Some(_) => {}
            None => self.kv_dim = Some(expected_dim),
        }
        self.keys.extend_from_slice(key.data());
        self.values.extend_from_slice(value.data());
        Ok(())
    }

    pub(super) fn append_batch(
        &mut self,
        key: &Tensor,
        value: &Tensor,
        layout: &AttentionLayout,
    ) -> Result<()> {
        let (key_rows, key_dim) = key.as_matrix()?;
        let (value_rows, value_dim) = value.as_matrix()?;
        let expected_dim = layout.num_key_value_heads * layout.head_dim;
        if key_rows == 0
            || key_rows != value_rows
            || key_dim != expected_dim
            || value_dim != expected_dim
        {
            return Err(InferError::Dimension(format!(
                "cache KV batch attend key/value [seq,{expected_dim}], reçu key={:?}, value={:?}",
                key.shape(),
                value.shape()
            )));
        }
        match self.kv_dim {
            Some(dim) if dim != expected_dim => {
                return Err(InferError::Dimension(format!(
                    "cache KV dim={dim} incompatible avec {expected_dim}"
                )))
            }
            Some(_) => {}
            None => self.kv_dim = Some(expected_dim),
        }
        self.keys.extend_from_slice(key.data());
        self.values.extend_from_slice(value.data());
        Ok(())
    }

    pub(super) fn len(&self) -> usize {
        match self.kv_dim {
            Some(dim) if dim > 0 => self.keys.len() / dim,
            _ => 0,
        }
    }

    pub(super) fn truncate(&mut self, len: usize) -> Result<()> {
        let Some(dim) = self.kv_dim else {
            if len == 0 {
                return Ok(());
            }
            return Err(InferError::Dimension(format!(
                "truncate KV CPU {len} sans dimension initialisée"
            )));
        };
        let new_len = len
            .checked_mul(dim)
            .ok_or_else(|| InferError::Dimension("truncate KV CPU déborde".to_string()))?;
        if new_len > self.keys.len() || new_len > self.values.len() {
            return Err(InferError::Dimension(format!(
                "truncate KV CPU {len} > len {}",
                self.len()
            )));
        }
        self.keys.truncate(new_len);
        self.values.truncate(new_len);
        Ok(())
    }
}
