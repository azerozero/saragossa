use super::{
    CLONE_DEFAULT_FRAMES_PER_TOKEN, CLONE_DEFAULT_MIN_FRAMES, CLONE_GENERATION_HARD_CAP,
    DEFAULT_REPEAT_FRAME_STOP,
};
use crate::{InferError, Result, Tensor};

pub(super) fn tts_generation_trace_enabled() -> bool {
    crate::runtime_flags::env_var("SARAGOSSA_TTS_TRACE_FRAMES")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

pub(super) fn tts_internal_profile_enabled() -> bool {
    crate::runtime_flags::env_var("SARAGOSSA_TTS_INTERNAL_PROFILE")
        .map(|value| {
            let value = value.trim();
            !value.is_empty() && value != "0" && !value.eq_ignore_ascii_case("false")
        })
        .unwrap_or(false)
}

/// Quatre trames représentent environ 320 ms d'audio à 12,5 Hz.
const DEFAULT_STREAM_LOT: usize = 4;

/// Taille fixe du lot incrémental, ou premier seuil du repli par préfixes.
/// Réglable via `SARAGOSSA_TTS_STREAM_LOT`, borné ≥ 1.
pub(super) fn tts_stream_first_lot() -> usize {
    crate::runtime_flags::env_var("SARAGOSSA_TTS_STREAM_LOT")
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_STREAM_LOT)
}

/// Arrêt anti-boucle du talker TTS : une frame codec complète identique répétée
/// plusieurs fois indique une dérive de décodage greedy, pas une prosodie utile.
/// `0` désactive le garde-fou pour les diagnostics.
pub(super) fn tts_repeat_frame_stop() -> usize {
    tts_repeat_frame_stop_from_env(
        crate::runtime_flags::env_var("SARAGOSSA_TTS_REPEAT_FRAME_STOP").as_deref(),
    )
}

pub(super) fn tts_repeat_frame_stop_from_env(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_REPEAT_FRAME_STOP)
}

pub(super) fn repeat_frame_stop_tripped(repeat_run: usize, threshold: usize) -> bool {
    threshold > 0 && repeat_run >= threshold
}

/// Table d'embedding consultée ligne à ligne sur CPU.
///
/// Une table bf16 reste en bf16 et n'est élargie qu'à la lecture des lignes
/// demandées : l'élargissement est exact, la mémoire résidente est divisée
/// par deux (622 Mo au lieu de 1,24 Go pour le vocabulaire Qwen3-TTS).
#[derive(Debug)]
pub(super) enum RowTable {
    F32(Tensor),
    Bf16 {
        rows: usize,
        cols: usize,
        bits: Vec<u16>,
    },
}

impl RowTable {
    pub(super) fn cols(&self) -> Result<usize> {
        match self {
            Self::F32(tensor) => Ok(tensor.as_matrix()?.1),
            Self::Bf16 { cols, .. } => Ok(*cols),
        }
    }

    pub(super) fn capacity_bytes(&self) -> usize {
        match self {
            Self::F32(tensor) => tensor.capacity_bytes(),
            Self::Bf16 { bits, .. } => bits.capacity() * size_of::<u16>(),
        }
    }

    pub(super) fn to_tensor(&self) -> Result<Tensor> {
        match self {
            Self::F32(tensor) => Ok(tensor.clone()),
            Self::Bf16 { rows, cols, bits } => Tensor::from_vec(
                vec![*rows, *cols],
                bits.iter().map(|&b| bf16_to_f32(b)).collect(),
            ),
        }
    }

    pub(super) fn gather_i32(&self, ids: &[i32]) -> Result<Tensor> {
        let (rows, cols, bits) = match self {
            Self::F32(tensor) => return gather_rows_i32(tensor, ids),
            Self::Bf16 { rows, cols, bits } => (*rows, *cols, bits),
        };
        if ids.is_empty() {
            return Err(InferError::Dimension("séquence de tokens vide".to_string()));
        }
        let mut out = Vec::with_capacity(ids.len() * cols);
        for id in ids {
            let row = usize::try_from(*id)
                .map_err(|_| InferError::Dimension(format!("id embedding négatif: {id}")))?;
            if row >= rows {
                return Err(InferError::Dimension(format!(
                    "token id {row} hors vocab {rows}"
                )));
            }
            out.extend(
                bits[row * cols..(row + 1) * cols]
                    .iter()
                    .map(|&b| bf16_to_f32(b)),
            );
        }
        Tensor::from_vec(vec![ids.len(), cols], out)
    }
}

fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

pub(super) fn gather_rows_i32(table: &Tensor, ids: &[i32]) -> Result<Tensor> {
    let mut rows = Vec::with_capacity(ids.len());
    for id in ids {
        let id = usize::try_from(*id)
            .map_err(|_| InferError::Dimension(format!("id embedding négatif: {id}")))?;
        rows.push(id);
    }
    crate::embed_tokens(table, &rows)
}

pub(super) fn push_rows(
    out: &mut Vec<f32>,
    tensor: &Tensor,
    start: usize,
    end: usize,
) -> Result<()> {
    if start > end || end > tensor.shape()[0] {
        return Err(InferError::Dimension(format!(
            "slice rows invalide [{start}..{end}] pour {:?}",
            tensor.shape()
        )));
    }
    for row in start..end {
        out.extend_from_slice(tensor.row_slice(row)?);
    }
    Ok(())
}

pub(super) fn add_into(left: &mut [f32], right: &[f32]) {
    for (left, right) in left.iter_mut().zip(right.iter()) {
        *left += *right;
    }
}

pub(super) fn usize_from_i32(value: i32, what: &str) -> Result<usize> {
    usize::try_from(value).map_err(|_| InferError::Config(format!("{what} négatif: {value}")))
}

pub(super) fn clone_effective_frame_cap(max_frames: usize, target_tokens: usize) -> usize {
    max_frames.min(CLONE_GENERATION_HARD_CAP).min(
        CLONE_DEFAULT_MIN_FRAMES.max(target_tokens.saturating_mul(CLONE_DEFAULT_FRAMES_PER_TOKEN)),
    )
}

pub(super) fn clone_sample_seed() -> u64 {
    crate::runtime_flags::env_var("SARAGOSSA_TTS_CLONE_SAMPLE_SEED")
        .and_then(|value| value.trim().parse::<u64>().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod row_table_tests {
    use super::*;

    #[test]
    fn bf16_rows_match_widened_f32_table_bitwise() -> Result<()> {
        let bits: Vec<u16> = vec![0x3f80, 0xbf00, 0x7f7f, 0x0001, 0x4049, 0xc2f7];
        let table = RowTable::Bf16 {
            rows: 3,
            cols: 2,
            bits: bits.clone(),
        };
        let dense = RowTable::F32(table.to_tensor()?);
        for ids in [&[2, 0, 1][..], &[1, 1]] {
            let got = table.gather_i32(ids)?;
            let want = dense.gather_i32(ids)?;
            assert_eq!(got.shape(), want.shape());
            let same = got
                .data()
                .iter()
                .zip(want.data())
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(same, "lignes bf16 différentes pour {ids:?}");
        }
        assert_eq!(table.to_tensor()?.data()[0], 1.0);
        assert_eq!(table.cols()?, 2);
        assert_eq!(table.capacity_bytes(), bits.capacity() * 2);
        assert!(table.gather_i32(&[3]).is_err());
        assert!(table.gather_i32(&[-1]).is_err());
        assert!(table.gather_i32(&[]).is_err());
        Ok(())
    }
}
