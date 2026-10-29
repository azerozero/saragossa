//! Lignes f32 ou bf16 lues en place dans un buffer partagé que le GPU a fini
//! d'écrire.

use super::*;

/// Élément d'une vue partagée : f32, ou bits bf16 (`u16`). Scellé : seuls des
/// types sans invalide ni padding peuvent être lus depuis la mémoire GPU.
pub(crate) trait SharedElement: Copy + sealed::Sealed {}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u16 {}
}

impl SharedElement for f32 {}
impl SharedElement for u16 {}

/// Vue CPU, sans copie, de `len` éléments d'un buffer `StorageModeShared` à
/// partir de l'élément `offset`. Le K/V d'un prefill complet y reste : sa
/// relecture vers un `Vec` neuf payait une faute de page par 16 Kio (382 ms à
/// 32k).
#[derive(Clone, Debug)]
pub(crate) struct SharedRows<T> {
    buffer: metal::Buffer,
    offset: usize,
    len: usize,
    element: std::marker::PhantomData<T>,
}

/// Lignes f32 (K/V ou hidden d'un prefill).
pub(crate) type SharedF32Rows = SharedRows<f32>;
/// Lignes bf16 stockées en bits `u16` (K/V d'un prefill stocké en bf16).
pub(crate) type SharedBf16Rows = SharedRows<u16>;

impl<T: SharedElement> SharedRows<T> {
    /// Fige `buffer[offset..offset + len]` (en éléments `T`).
    ///
    /// # Safety
    ///
    /// `buffer[offset..offset + len]` doit être initialisé et ses écritures
    /// GPU terminées ; ensuite plus aucune écriture, GPU ou CPU, ne doit le
    /// viser (les lectures GPU, semis de l'arène, restent permises) : la vue
    /// en expose le contenu en `&[T]` à durée de vie libre côté lecteur. Seules les lignes qu'aucune vue
    /// vivante du buffer (clones et troncatures compris) ne couvre peuvent
    /// encore être écrites (prolongement en place d'un prefill dans ses lignes
    /// réservées).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la plage dépasse le buffer ou si le buffer n'a
    /// pas de pointeur CPU (stockage privé).
    #[expect(
        unsafe_code,
        reason = "contrat d'immuabilité du buffer reporté sur l'appelant"
    )]
    pub(crate) unsafe fn new(buffer: metal::Buffer, offset: usize, len: usize) -> Result<Self> {
        let end = offset
            .checked_add(len)
            .and_then(|end| end.checked_mul(std::mem::size_of::<T>()))
            .ok_or_else(|| InferError::Dimension("vue partagée déborde".to_string()))?;
        if u64::try_from(end).map_or(true, |end| end > buffer.length()) {
            return Err(InferError::Dimension(format!(
                "vue partagée {offset}+{len} éléments hors du buffer de {} octets",
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
            element: std::marker::PhantomData,
        })
    }

    #[expect(
        unsafe_code,
        reason = "lecture CPU en place d'un MTLBuffer StorageModeShared figé"
    )]
    pub(crate) fn as_slice(&self) -> &[T] {
        // SAFETY: `new` a vérifié que `offset + len` éléments tiennent dans le
        // buffer partagé et que son pointeur CPU existe (aligné page, donc
        // aligné pour `T`, f32 ou u16 scellés : toute suite d'octets est une
        // valeur valide). Plus
        // rien n'écrit cette plage (contrat `# Safety` de `new`) et les vues ne
        // donnent qu'un accès en lecture : aucune écriture concurrente.
        unsafe {
            std::slice::from_raw_parts(
                self.buffer.contents().cast::<T>().add(self.offset),
                self.len,
            )
        }
    }

    /// Buffer source, à ne lier qu'en lecture (copie GPU vers l'arène).
    pub(crate) fn buffer(&self) -> &BufferRef {
        &self.buffer
    }

    /// Identité de l'allocation, partagée par toutes ses vues (troncatures et
    /// clones compris) : le comptage mémoire n'en retient qu'une.
    pub(crate) fn allocation_id(&self) -> usize {
        self.buffer.contents().addr()
    }

    /// Taille entière de l'allocation en octets, qu'une vue tronquée garde en vie.
    pub(crate) fn allocation_bytes(&self) -> usize {
        usize::try_from(self.buffer.length()).unwrap_or(usize::MAX)
    }

    pub(crate) fn byte_offset(&self) -> u64 {
        // Borné par `buffer.length()` à la construction.
        (self.offset * std::mem::size_of::<T>()) as u64
    }

    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Raccourcit la vue ; le buffer reste alloué tant qu'une vue le tient.
    pub(crate) fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

/// K, V `[rows, kv_dim]` ou hidden final `[rows, hidden]` rendu par le prefill
/// résident : vue du buffer GPU qu'il a seul écrit, ou copie relue (extension
/// de K/V dont le buffer porte aussi le préfixe, dernière ligne seule). Le K/V
/// d'un prefill stocké en bf16 arrive en bits `u16` ; le hidden reste f32.
#[derive(Clone, Debug)]
pub(crate) enum PrefillKvRows {
    Owned(Vec<f32>),
    OwnedBf16(Vec<u16>),
    Shared(SharedF32Rows),
    SharedBf16(SharedBf16Rows),
}

impl PrefillKvRows {
    /// Nombre d'éléments, sans lecture des données.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Owned(data) => data.len(),
            Self::OwnedBf16(data) => data.len(),
            Self::Shared(rows) => rows.len(),
            Self::SharedBf16(rows) => rows.len(),
        }
    }

    /// Segment typé, vue partagée comprise (préfixe d'un prolongement).
    #[cfg(test)]
    pub(crate) fn segment(&self) -> crate::decode_resident::KvSegment<'_> {
        use crate::decode_resident::KvSegment;
        match self {
            Self::Owned(data) => KvSegment::Cpu(data),
            Self::OwnedBf16(data) => KvSegment::CpuBf16(data),
            Self::Shared(rows) => KvSegment::Shared(rows),
            Self::SharedBf16(rows) => KvSegment::SharedBf16(rows),
        }
    }

    /// Tenseur `[len / cols, cols]` ; une vue partagée est copiée.
    pub(crate) fn into_tensor(self, cols: usize) -> Result<Tensor> {
        let data = match self {
            Self::Owned(data) => data,
            Self::Shared(rows) => rows.as_slice().to_vec(),
            Self::OwnedBf16(_) | Self::SharedBf16(_) => {
                return Err(InferError::Dimension(
                    "lignes de prefill bf16 en tenseur f32".to_string(),
                ))
            }
        };
        Tensor::from_vec(vec![data.len() / cols.max(1), cols], data)
    }
}

#[cfg(test)]
impl MetalExecutor {
    /// Vue partagée d'une copie de `data`, comme le hidden d'un prefill.
    #[expect(unsafe_code, reason = "buffer neuf propre au test")]
    pub(crate) fn shared_f32_rows_for_test(&self, data: &[f32]) -> Result<SharedF32Rows> {
        let buffer = self.uncached_f32_buffer(data.len(), "test_shared_rows")?;
        write_f32_buffer(&buffer, data)?;
        // SAFETY: buffer neuf, hors scratch, que plus rien n'écrit ensuite.
        unsafe { SharedF32Rows::new(buffer, 0, data.len()) }
    }
}
