//! Format du KV résident full-attn : dense (f32, bf16) ou quantifié affine.

use super::GpuElement;
use crate::{InferError, Result};
use rayon::prelude::*;

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

    /// Bits par code, `None` pour un encodage dense.
    pub(crate) const fn bits(self) -> Option<usize> {
        match self {
            Self::F32 | Self::Bf16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }

    /// Éléments du buffer de charge utile pour `cells` valeurs (octets pour un
    /// encodage quantifié), `None` si `cells` ne remplit pas des groupes
    /// entiers ou déborde.
    pub(crate) fn payload_len(self, cells: usize) -> Option<usize> {
        match self.bits() {
            None => Some(cells),
            Some(bits) if cells % KV_QUANT_GROUP == 0 => {
                cells.checked_mul(bits).map(|total| total / 8)
            }
            Some(_) => None,
        }
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

/// Valeurs par groupe de quantification, contiguës le long de `head_dim`
/// (d256 : 4 groupes par tête et par ligne, d128 : 2). Un groupe ne chevauche
/// jamais deux têtes puisque `head_dim` est un multiple de 64.
pub(crate) const KV_QUANT_GROUP: usize = 64;

/// Code affine d'une valeur, miroir exact de `kv_code` du kernel : échelle
/// nulle ou NaN, valeur NaN ou sous le biais → 0 ; saturation à `levels`. Les
/// comparaisons explicites rendent ces cas définis à l'identique sur CPU et GPU.
fn quant_code(x: f32, scale: f32, bias: f32, levels: u8) -> u8 {
    let t = (x - bias) / scale;
    if scale > 0.0 && t > 0.0 {
        if t >= f32::from(levels) {
            levels
        } else {
            // `t` est dans ]0, levels[ : l'arrondi tient dans un u8.
            t.round_ties_even() as u8
        }
    } else {
        0
    }
}

/// Quantifie un groupe de [`KV_QUANT_GROUP`] valeurs : écrit les codes
/// empaquetés dans `out` (64 octets en Q8, 32 en Q4, `q[2i]` dans le quartet
/// bas) et renvoie `[échelle, biais]`.
///
/// min/max séquentiels de `j = 0` à 63 avec comparaisons strictes, comme le
/// kernel : un NaN en tête se propage à l'échelle (codes tous nuls), ailleurs
/// il est ignoré par les bornes et codé 0.
fn quantize_group(bits: usize, x: &[f32], out: &mut [u8]) -> [f32; 2] {
    let levels: u8 = if bits == 8 { 255 } else { 15 };
    let mut lo = x[0];
    let mut hi = x[0];
    for &value in &x[1..] {
        if value < lo {
            lo = value;
        }
        if value > hi {
            hi = value;
        }
    }
    let scale = (hi - lo) / f32::from(levels);
    if bits == 8 {
        for (code, &value) in out.iter_mut().zip(x) {
            *code = quant_code(value, scale, lo, levels);
        }
    } else {
        for (byte, pair) in out.iter_mut().zip(x.chunks_exact(2)) {
            *byte = quant_code(pair[0], scale, lo, levels)
                | (quant_code(pair[1], scale, lo, levels) << 4);
        }
    }
    [scale, lo]
}

/// Vérifie les longueurs d'un bloc de lignes quantifiées et renvoie les bits
/// par code.
fn check_quant_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    values: usize,
    payload: usize,
    scales: usize,
) -> Result<usize> {
    let bits = encoding
        .bits()
        .ok_or_else(|| InferError::Config(format!("KV quantifié: encodage {encoding:?} dense")))?;
    if kv_dim == 0 || kv_dim % KV_QUANT_GROUP != 0 || values % kv_dim != 0 {
        return Err(InferError::Dimension(format!(
            "KV quantifié: {values} valeurs pour des lignes de {kv_dim} (groupes de {KV_QUANT_GROUP})"
        )));
    }
    let groups = values / KV_QUANT_GROUP;
    if encoding.payload_len(values) != Some(payload) || groups.checked_mul(2) != Some(scales) {
        return Err(InferError::Dimension(format!(
            "KV quantifié: {values} valeurs, charge utile {payload} octets, {scales} échelles"
        )));
    }
    Ok(bits)
}

/// Oracle CPU : quantifie des lignes `[rows, kv_dim]` vers `payload` (codes)
/// et `scales` (`[rows, kv_dim / 64]` paires échelle, biais), même disposition
/// que le KV résident. Parallèle par groupe, chacun étant indépendant.
///
/// # Errors
///
/// Renvoie une erreur si l'encodage est dense ou si les longueurs ne forment
/// pas des lignes entières de groupes complets.
pub(super) fn quantize_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    values: &[f32],
    payload: &mut [u8],
    scales: &mut [f32],
) -> Result<()> {
    let bits = check_quant_rows(encoding, kv_dim, values.len(), payload.len(), scales.len())?;
    values
        .par_chunks(KV_QUANT_GROUP)
        .zip(payload.par_chunks_mut(KV_QUANT_GROUP * bits / 8))
        .zip(scales.par_chunks_mut(2))
        .for_each(|((x, out), scale_bias)| {
            scale_bias.copy_from_slice(&quantize_group(bits, x, out));
        });
    Ok(())
}

/// Oracle CPU : déquantifie des lignes produites par [`quantize_rows`] (ou
/// par le kernel) vers `out`, avec `fma(code, échelle, biais)` comme le kernel
/// d'attention : les valeurs relues sont exactement celles que lit le GPU.
///
/// # Errors
///
/// Mêmes conditions que [`quantize_rows`].
pub(super) fn dequantize_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    payload: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let bits = check_quant_rows(encoding, kv_dim, out.len(), payload.len(), scales.len())?;
    out.par_chunks_mut(KV_QUANT_GROUP)
        .zip(payload.par_chunks(KV_QUANT_GROUP * bits / 8))
        .zip(scales.par_chunks(2))
        .for_each(|((x, codes), scale_bias)| {
            let (scale, bias) = (scale_bias[0], scale_bias[1]);
            for (index, value) in x.iter_mut().enumerate() {
                let code = if bits == 8 {
                    codes[index]
                } else {
                    (codes[index / 2] >> ((index % 2) * 4)) & 0x0f
                };
                *value = f32::from(code).mul_add(scale, bias);
            }
        });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize(encoding: KvEncoding, kv_dim: usize, values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let bytes = encoding.payload_len(values.len()).unwrap_or(0);
        let mut payload = vec![0_u8; bytes];
        let mut scales = vec![0.0_f32; values.len() / KV_QUANT_GROUP * 2];
        quantize_rows(encoding, kv_dim, values, &mut payload, &mut scales)
            .expect("invariant: longueurs de test cohérentes");
        (payload, scales)
    }

    fn roundtrip(encoding: KvEncoding, kv_dim: usize, values: &[f32]) -> Vec<f32> {
        let (payload, scales) = quantize(encoding, kv_dim, values);
        let mut out = vec![0.0_f32; values.len()];
        dequantize_rows(encoding, kv_dim, &payload, &scales, &mut out)
            .expect("invariant: longueurs de test cohérentes");
        out
    }

    #[test]
    fn aller_retour_borne_par_demi_pas() {
        let kv_dim = 256;
        let values: Vec<f32> = (0..4 * kv_dim)
            .map(|i| (i as f32 * 0.731).sin() * (1.0 + (i / 64) as f32))
            .collect();
        for (encoding, levels) in [(KvEncoding::Q8, 255.0_f32), (KvEncoding::Q4, 15.0)] {
            let out = roundtrip(encoding, kv_dim, &values);
            for (group, (x, y)) in values
                .chunks(KV_QUANT_GROUP)
                .zip(out.chunks(KV_QUANT_GROUP))
                .enumerate()
            {
                let lo = x.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let half_step = (hi - lo) / levels / 2.0;
                for (a, b) in x.iter().zip(y) {
                    assert!(
                        (a - b).abs() <= half_step * 1.001 + 1e-6,
                        "{encoding:?} groupe {group}: {a} -> {b} (demi-pas {half_step})"
                    );
                }
                // Le minimum du groupe est codé 0 : restitué exactement.
                assert_eq!(y.iter().copied().fold(f32::INFINITY, f32::min), lo);
            }
        }
    }

    #[test]
    fn groupe_constant_et_disposition_q4() {
        let mut values = vec![3.5_f32; 128];
        // Second groupe : 0..15 répétés → codes exacts, quartet bas = indice pair.
        for (i, value) in values[64..].iter_mut().enumerate() {
            *value = (i % 16) as f32;
        }
        let (payload, scales) = quantize(KvEncoding::Q4, 64, &values);
        assert_eq!(&scales, &[0.0, 3.5, 1.0, 0.0], "échelle nulle, biais exact");
        assert!(payload[..32].iter().all(|&b| b == 0), "groupe constant → 0");
        assert_eq!(payload[32], 0x10, "q[0]=0 bas, q[1]=1 haut");
        assert_eq!(payload[39], 0xfe, "q[14]=14 bas, q[15]=15 haut");
        assert_eq!(
            roundtrip(KvEncoding::Q4, 64, &values),
            values,
            "valeurs sur la grille : exactes"
        );
    }

    #[test]
    fn valeurs_non_finies_definies() {
        let mut values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        values[5] = f32::NAN;
        // NaN hors tête : ignoré par les bornes, codé 0.
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert_eq!(scales, [63.0 / 255.0, 0.0]);
        assert_eq!(payload[5], 0);
        assert_eq!(payload[63], 255);
        // NaN en tête : échelle NaN, tous les codes à 0.
        values[0] = f32::NAN;
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert!(scales[0].is_nan());
        assert!(payload.iter().all(|&b| b == 0));
        // Infini : échelle infinie, codes définis (0), aucun code hors plage.
        values[0] = f32::INFINITY;
        values[5] = 1.0;
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert_eq!(scales, [f32::INFINITY, 1.0]);
        assert!(payload.iter().all(|&b| b == 0));
    }

    #[test]
    fn longueurs_incoherentes_refusees() {
        let mut payload = vec![0_u8; 64];
        let mut scales = vec![0.0_f32; 2];
        let values = vec![0.0_f32; 64];
        assert!(quantize_rows(KvEncoding::Bf16, 64, &values, &mut payload, &mut scales).is_err());
        assert!(quantize_rows(KvEncoding::Q8, 96, &values, &mut payload, &mut scales).is_err());
        assert!(quantize_rows(KvEncoding::Q4, 64, &values, &mut payload, &mut scales).is_err());
        assert_eq!(KvEncoding::Q4.payload_len(64), Some(32));
        assert_eq!(KvEncoding::Q8.payload_len(32), None);
        assert_eq!(KvEncoding::Bf16.payload_len(3), Some(3));
    }
}
