//! Format du KV résident full-attn : dense (f32, bf16) ou quantifié affine.

use super::GpuElement;

/// Encodage d'UN tenseur résident K ou V.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvEncoding {
    F32,
    Bf16,
    /// Codes 8 bits, échelle et biais par groupe de valeurs.
    Q8,
    /// Codes 4 bits, deux par octet, échelle et biais par groupe de valeurs.
    Q4,
}

impl KvEncoding {
    /// Type d'élément du buffer de charge utile.
    pub(crate) const fn element(self) -> GpuElement {
        match self {
            Self::F32 => GpuElement::F32,
            Self::Bf16 => GpuElement::Bf16,
            Self::Q8 | Self::Q4 => GpuElement::U8,
        }
    }

    /// Indique un encodage quantifié (codes + échelles).
    pub(crate) const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::Q4)
    }
}

/// Format du KV résident d'une couche full-attn.
///
/// K et V portent chacun leur encodage : une erreur sur K déforme les scores
/// avant le softmax, une erreur sur V seulement la moyenne pondérée, d'où le
/// mode mixte K8V4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvFormat {
    F32,
    Bf16,
    Q8,
    K8V4,
    Q4,
}

impl KvFormat {
    /// Encodage des clés.
    pub(crate) const fn keys(self) -> KvEncoding {
        match self {
            Self::F32 => KvEncoding::F32,
            Self::Bf16 => KvEncoding::Bf16,
            Self::Q8 | Self::K8V4 => KvEncoding::Q8,
            Self::Q4 => KvEncoding::Q4,
        }
    }

    /// Encodage des valeurs.
    pub(crate) const fn values(self) -> KvEncoding {
        match self {
            Self::F32 => KvEncoding::F32,
            Self::Bf16 => KvEncoding::Bf16,
            Self::Q8 => KvEncoding::Q8,
            Self::K8V4 | Self::Q4 => KvEncoding::Q4,
        }
    }

    /// Indique un format quantifié (K et V quantifiés ensemble).
    pub(crate) const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::K8V4 | Self::Q4)
    }
}
