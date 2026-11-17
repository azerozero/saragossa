//! Insertion stricte et positions Qwen M-RoPE (fusion spatiale 2 × 2).

#[cfg(feature = "vision")]
use crate::Tensor;
use crate::{InferError, Result};
#[cfg(feature = "vision")]
use sha2::{Digest, Sha256};

/// Les paramètres sont sérialisés canoniquement par le producteur : resize,
/// normalisation, timestamps éventuels et identité des poids vision compris.
#[cfg(feature = "vision")]
pub(crate) struct SuppliedImage<'a> {
    pub start: usize,
    pub grid: [usize; 3],
    pub embeddings: &'a Tensor,
    pub content_hash: [u8; 32],
    pub parameters: &'a [u8],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct MediaSpan {
    pub start: usize,
    pub end: usize,
    pub hash: [u8; 32],
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub(crate) struct VisionPositions {
    pub rows: Vec<[usize; 3]>,
    pub media: Vec<MediaSpan>,
}

impl VisionPositions {
    pub(crate) fn estimated_bytes(&self) -> usize {
        self.rows
            .capacity()
            .saturating_mul(std::mem::size_of::<[usize; 3]>())
            .saturating_add(
                self.media
                    .capacity()
                    .saturating_mul(std::mem::size_of::<MediaSpan>()),
            )
    }

    /// Conserve uniquement le contexte déjà calculé à une frontière sûre.
    pub(crate) fn prefix(&self, tokens: usize) -> Result<Option<Self>> {
        self.delta_at(tokens)?;
        let media: Vec<_> = self
            .media
            .iter()
            .filter(|m| m.end <= tokens)
            .cloned()
            .collect();
        Ok((!media.is_empty()).then(|| Self {
            rows: self.rows[..tokens].to_vec(),
            media,
        }))
    }

    /// Une frontière interne à une image n'est pas un préfixe réutilisable.
    pub(crate) fn delta_at(&self, tokens: usize) -> Result<i64> {
        let mut previous_end = 0;
        for media in &self.media {
            if media.start < previous_end || media.start >= media.end || media.end > self.rows.len()
            {
                return Err(invalid("spans du contexte positionnel incohérents"));
            }
            previous_end = media.end;
        }
        if tokens > self.rows.len()
            || self
                .media
                .iter()
                .any(|m| m.start < tokens && tokens < m.end)
        {
            return Err(invalid("frontière interne au média ou hors prompt"));
        }
        let next = self.rows[..tokens]
            .iter()
            .flatten()
            .max()
            .copied()
            .map_or(Some(0), |p| p.checked_add(1))
            .ok_or_else(|| invalid("position trop grande"))?;
        Ok(
            i64::try_from(next).map_err(|_| invalid("position hors i64"))?
                - i64::try_from(tokens).map_err(|_| invalid("tokens hors i64"))?,
        )
    }
}

fn invalid(message: &str) -> InferError {
    InferError::Dimension(format!("vision: {message}"))
}

/// Valide tout avant de copier ; aucune insertion partielle en cas d'erreur.
#[cfg(feature = "vision")]
pub(crate) fn insert_embeddings(
    tokens: &[usize],
    image_pad: usize,
    text: &Tensor,
    images: &[SuppliedImage<'_>],
) -> Result<(Tensor, VisionPositions)> {
    let (seq, width) = text.as_matrix()?;
    if seq != tokens.len() || width == 0 || images.is_empty() {
        return Err(invalid("prompt, largeur ou médias vides/incompatibles"));
    }
    let mut previous_end = 0;
    let mut count = 0usize;
    let mut media = Vec::with_capacity(images.len());
    for image in images {
        let [t, h, w] = image.grid;
        if t != 1 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
            return Err(invalid("image attendue T=1, H/W positifs et pairs"));
        }
        let rows = t
            .checked_mul(h / 2)
            .and_then(|n| n.checked_mul(w / 2))
            .ok_or_else(|| invalid("grille trop grande"))?;
        let end = image
            .start
            .checked_add(rows)
            .ok_or_else(|| invalid("span trop grand"))?;
        if image.start < previous_end
            || end > seq
            || image.embeddings.as_matrix()? != (rows, width)
            || !image.embeddings.data().iter().all(|v| v.is_finite())
            || tokens[image.start..end].iter().any(|&id| id != image_pad)
        {
            return Err(invalid(
                "ordre, placeholders ou lignes d'embeddings incompatibles",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"saragossa-vision-v1\0");
        hash.update(image.content_hash);
        for dim in image.grid {
            hash.update((dim as u64).to_le_bytes());
        }
        hash.update((image.parameters.len() as u64).to_le_bytes());
        hash.update(image.parameters);
        // L'API interne accepte des embeddings fournis : leur contenu fait
        // aussi partie de l'identité, même si les métadonnées sont erronées.
        for v in image.embeddings.data() {
            hash.update(v.to_le_bytes());
        }
        media.push(MediaSpan {
            start: image.start,
            end,
            hash: hash.finalize().into(),
        });
        count = count
            .checked_add(rows)
            .ok_or_else(|| invalid("compte trop grand"))?;
        previous_end = end;
    }
    if tokens.iter().filter(|&&id| id == image_pad).count() != count {
        return Err(invalid(
            "nombre de placeholders différent du nombre de lignes vision",
        ));
    }
    let mut hidden = text.data().to_vec();
    let mut positions = Vec::with_capacity(seq);
    let mut cursor = 0usize;
    for (image, span) in images.iter().zip(&media) {
        while positions.len() < span.start {
            positions.push([cursor; 3]);
            cursor += 1;
        }
        let [_, h, w] = image.grid;
        for y in 0..h / 2 {
            for x in 0..w / 2 {
                positions.push([cursor, cursor + y, cursor + x]);
            }
        }
        cursor += (h / 2).max(w / 2);
        hidden[span.start * width..span.end * width].copy_from_slice(image.embeddings.data());
    }
    while positions.len() < seq {
        positions.push([cursor; 3]);
        cursor += 1;
    }
    Ok((
        Tensor::from_vec(vec![seq, width], hidden)?,
        VisionPositions {
            rows: positions,
            media,
        },
    ))
}

#[cfg(test)]
#[cfg(feature = "vision")]
mod tests;
