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
        self.check_batch(key, value, layout)?;
        self.keys.extend_from_slice(key.data());
        self.values.extend_from_slice(value.data());
        Ok(())
    }

    /// Comme [`Self::append_batch`], en reprenant les données sans copie : le
    /// KV relu du prefill devient un segment du cache.
    pub(super) fn append_batch_owned(
        &mut self,
        key: Tensor,
        value: Tensor,
        layout: &AttentionLayout,
    ) -> Result<()> {
        self.check_batch(&key, &value, layout)?;
        self.keys.push_owned(key.into_data());
        self.values.push_owned(value.into_data());
        Ok(())
    }

    /// Ajoute le K/V `[rows, kv_dim]` d'un prefill résident, vues GPU
    /// partagées comprises, sans recopie. `base` > 0 (couche glissante
    /// bornée à sa fenêtre) : `base` lignes absentes précèdent `key`/`value`,
    /// sur un cache vide seulement.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn append_prefill(
        &mut self,
        key: crate::metal_backend::PrefillKvRows,
        value: crate::metal_backend::PrefillKvRows,
        kv_dim: usize,
        base: usize,
        layout: &AttentionLayout,
    ) -> Result<()> {
        let expected_dim = layout.num_key_value_heads * layout.head_dim;
        let (key_len, value_len) = (key.len(), value.len());
        let base_len = base.checked_mul(expected_dim);
        if expected_dim == 0
            || kv_dim != expected_dim
            || key_len == 0
            || key_len != value_len
            || key_len % expected_dim != 0
            || self.kv_dim.is_some_and(|dim| dim != expected_dim)
            || base_len.is_none()
            || (base > 0 && self.keys.len() + self.values.len() > 0)
        {
            return Err(InferError::Dimension(format!(
                "cache KV prefill attend [seq,{expected_dim}] (base {base} sur cache vide), reçu key={key_len} value={value_len} (dim courante {:?}, {} lignes)",
                self.kv_dim,
                self.len()
            )));
        }
        self.kv_dim = Some(expected_dim);
        if let Some(base_len) = base_len.filter(|len| *len > 0) {
            let accepted = self.keys.set_base(base_len) && self.values.set_base(base_len);
            accepted
                .then_some(())
                .expect("invariant: base posée sur un cache vide, vérifié ci-dessus");
        }
        self.keys.push_prefill(key);
        self.values.push_prefill(value);
        Ok(())
    }

    /// Remplace tout le K/V par celui d'un prolongement résident (`rows`
    /// lignes, préfixe compris, dont les `base` premières absentes),
    /// recopié dans un buffer neuf ou prolongé en place dans le sien, validé
    /// avant toute mutation.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn replace_prefill(
        &mut self,
        key: crate::metal_backend::PrefillKvRows,
        value: crate::metal_backend::PrefillKvRows,
        kv_dim: usize,
        (rows, base): (usize, usize),
        layout: &AttentionLayout,
    ) -> Result<()> {
        let expected = rows
            .checked_sub(base)
            .and_then(|stored| stored.checked_mul(kv_dim));
        if expected != Some(key.len())
            || expected != Some(value.len())
            || kv_dim != layout.num_key_value_heads * layout.head_dim
            || self.kv_dim.is_some_and(|dim| dim != kv_dim)
        {
            return Err(InferError::Dimension(format!(
                "K/V prolongé recopié: {}/{} au lieu de ({rows} - {base})×{kv_dim}",
                key.len(),
                value.len()
            )));
        }
        // Tout ce qu'`append_prefill` vérifie l'a été ci-dessus : vider est sûr.
        self.keys = Default::default();
        self.values = Default::default();
        self.append_prefill(key, value, kv_dim, base, layout)
    }

    /// Lignes absentes devant le K/V stocké (couche glissante bornée à sa
    /// fenêtre) : aucune lecture ni troncature ne descend en dessous.
    pub(super) fn base_rows(&self) -> usize {
        match self.kv_dim {
            Some(dim) if dim > 0 => self.keys.base() / dim,
            _ => 0,
        }
    }

    fn check_batch(
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
        // Lignes sorties de la fenêtre : elles ne peuvent pas revenir.
        if new_len < self.keys.base() || new_len < self.values.base() {
            return Err(InferError::Dimension(format!(
                "truncate KV CPU {len} sous la base {} (fenêtre bornée)",
                self.base_rows()
            )));
        }
        self.keys.truncate(new_len);
        self.values.truncate(new_len);
        Ok(())
    }
}
