//! Lignes f32 lues en place dans un buffer partagé que le GPU a fini d'écrire.

use super::*;

/// Vue CPU, sans copie, de `len` f32 d'un buffer `StorageModeShared` à partir
/// de l'élément `offset`. Le K/V d'un prefill complet y reste : sa relecture
/// vers un `Vec` neuf payait une faute de page par 16 Kio (382 ms à 32k).
#[derive(Clone, Debug)]
pub(crate) struct SharedF32Rows {
    buffer: metal::Buffer,
    offset: usize,
    len: usize,
}

impl SharedF32Rows {
    /// Fige `buffer[offset..offset + len]`. L'appelant garantit que plus
    /// aucune commande GPU n'écrira ce buffer : la vue n'est relue que par le
    /// CPU et par des copies GPU en lecture.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la plage dépasse le buffer ou si le buffer n'a
    /// pas de pointeur CPU (stockage privé).
    pub(crate) fn new(buffer: metal::Buffer, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .and_then(|end| end.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| InferError::Dimension("vue partagée déborde".to_string()))?;
        if u64::try_from(end).map_or(true, |end| end > buffer.length()) {
            return Err(InferError::Dimension(format!(
                "vue partagée {offset}+{len} f32 hors du buffer de {} octets",
                buffer.length()
            )));
        }
        if buffer.contents().is_null() {
            return Err(InferError::Metal(
                "vue partagée sur un buffer sans pointeur CPU".to_string(),
            ));
        }
        Ok(Self {
            buffer,
            offset,
            len,
        })
    }

    #[expect(
        unsafe_code,
        reason = "lecture CPU en place d'un MTLBuffer StorageModeShared figé"
    )]
    pub(crate) fn as_slice(&self) -> &[f32] {
        // SAFETY: `new` a vérifié que `offset + len` f32 tiennent dans le buffer
        // partagé et que son pointeur CPU existe (aligné page, donc f32). Plus
        // aucune commande GPU ne l'écrit (contrat de `new`) et les vues ne
        // donnent qu'un accès en lecture : aucune écriture concurrente.
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<f32>().add(self.offset),
                self.len,
            )
        }
    }

    pub(crate) fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    pub(crate) fn byte_offset(&self) -> u64 {
        // Borné par `buffer.length()` à la construction.
        (self.offset * std::mem::size_of::<f32>()) as u64
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Raccourcit la vue ; le buffer reste alloué tant qu'une vue le tient.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

/// K ou V `[rows, kv_dim]` rendu par le prefill résident : vue du buffer GPU
/// pour un prefill complet, copie relue pour une extension (son buffer porte
/// aussi le préfixe, déjà dans le cache CPU).
#[derive(Clone, Debug)]
pub(crate) enum PrefillKvRows {
    Owned(Vec<f32>),
    Shared(SharedF32Rows),
}

impl PrefillKvRows {
    pub(crate) fn data(&self) -> &[f32] {
        match self {
            Self::Owned(data) => data,
            Self::Shared(rows) => rows.as_slice(),
        }
    }
}
