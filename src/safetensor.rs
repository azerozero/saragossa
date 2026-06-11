//! Lecture typée des tenseurs stockés en safetensors.

use crate::{InferError, Result, Tensor};
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Charge un tenseur f32 nommé depuis un fichier safetensors.
///
/// # Errors
///
/// Renvoie une erreur si la lecture, le nom ou le dtype échoue.
pub fn load_f32_tensor(path: impl AsRef<Path>, name: &str) -> Result<Tensor> {
    with_view(path.as_ref(), name, |view| {
        if view.dtype() != Dtype::F32 {
            return Err(InferError::UnsupportedDtype {
                name: name.to_string(),
                dtype: view.dtype(),
            });
        }
        tensor_from_safetensor_parts(name, view.dtype(), view.shape(), view.data())
    })
}

/// Charge un tenseur flottant nommé depuis un fichier safetensors.
///
/// # Errors
///
/// Renvoie une erreur si la lecture, le nom ou le dtype échoue.
pub fn load_float_tensor(path: impl AsRef<Path>, name: &str) -> Result<Tensor> {
    with_view(path.as_ref(), name, |view| {
        tensor_from_safetensor_parts(name, view.dtype(), view.shape(), view.data())
    })
}

fn with_view<R>(path: &Path, name: &str, f: impl FnOnce(TensorView<'_>) -> Result<R>) -> Result<R> {
    let bytes = read_file(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| InferError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;
    let view = tensors
        .tensor(name)
        .map_err(|_| InferError::MissingWeight(name.to_string()))?;
    f(view)
}

/// Charge tous les tenseurs f32 d'un fichier safetensors.
///
/// # Errors
///
/// Renvoie une erreur si la lecture ou une conversion échoue.
pub fn load_f32_tensors(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| InferError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = HashMap::new();
    for (name, view) in tensors.iter() {
        if view.dtype() == Dtype::F32 {
            out.insert(
                name.to_string(),
                Tensor::from_vec(view.shape().to_vec(), bytes_to_f32(view.data(), name)?)?,
            );
        }
    }
    Ok(out)
}

/// Charge tous les tenseurs flottants d'un fichier safetensors.
///
/// # Errors
///
/// Renvoie une erreur si la lecture ou une conversion échoue.
pub fn load_float_tensors(path: impl AsRef<Path>) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| InferError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = HashMap::new();
    for (name, view) in tensors.iter() {
        if is_dense_float_dtype(view.dtype()) {
            out.insert(
                name.to_string(),
                tensor_from_safetensor_parts(name, view.dtype(), view.shape(), view.data())?,
            );
        }
    }
    Ok(out)
}

/// Charge les tenseurs flottants et applique les échelles FP8 disponibles.
///
/// # Errors
///
/// Renvoie une erreur si la lecture, la conversion ou l'échelle échoue.
pub fn load_float_tensors_with_fp8_scale_inv(
    path: impl AsRef<Path>,
) -> Result<HashMap<String, Tensor>> {
    let path = path.as_ref();
    let bytes = read_file(path)?;
    let tensors = SafeTensors::deserialize(&bytes).map_err(|source| InferError::Safetensors {
        path: path.to_path_buf(),
        source,
    })?;
    let mut out = HashMap::new();
    for (name, view) in tensors.iter() {
        if name.ends_with(".weight_scale_inv") || !is_dense_float_dtype(view.dtype()) {
            continue;
        }
        let mut tensor =
            tensor_from_safetensor_parts(name, view.dtype(), view.shape(), view.data())?;
        if is_fp8_weight(view.dtype(), name) {
            let scale_key = format!(
                "{}.weight_scale_inv",
                name.strip_suffix(".weight").ok_or_else(|| {
                    InferError::Config(format!("poids FP8 sans suffixe .weight: {name}"))
                })?
            );
            if let Ok(scale_view) = tensors.tensor(&scale_key) {
                let scales = bytes_to_dense_f32(scale_view.data(), scale_view.dtype(), &scale_key)?;
                tensor = apply_fp8_scales(tensor, &scales, scale_view.shape(), &scale_key, 128)?;
            }
        }
        out.insert(name.to_string(), tensor);
    }
    Ok(out)
}

pub(crate) fn tensor_from_safetensor_parts(
    name: &str,
    dtype: Dtype,
    shape: &[usize],
    data: &[u8],
) -> Result<Tensor> {
    Tensor::from_vec(shape.to_vec(), bytes_to_dense_f32(data, dtype, name)?)
}

pub(crate) fn bytes_to_dense_f32(bytes: &[u8], dtype: Dtype, name: &str) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => bytes_to_f32(bytes, name),
        Dtype::BF16 => bytes_to_bf16_f32(bytes, name),
        Dtype::F16 => bytes_to_f16_f32(bytes, name),
        Dtype::F8_E4M3 => bytes_to_f8_e4m3_f32(bytes),
        Dtype::F8_E5M2 => bytes_to_f8_e5m2_f32(bytes),
        _ => Err(InferError::UnsupportedDtype {
            name: name.to_string(),
            dtype,
        }),
    }
}

fn is_dense_float_dtype(dtype: Dtype) -> bool {
    matches!(
        dtype,
        Dtype::F32 | Dtype::BF16 | Dtype::F16 | Dtype::F8_E4M3 | Dtype::F8_E5M2
    )
}

fn is_fp8_weight(dtype: Dtype, name: &str) -> bool {
    matches!(dtype, Dtype::F8_E4M3 | Dtype::F8_E5M2) && name.ends_with(".weight")
}

fn apply_fp8_scales(
    tensor: Tensor,
    scales: &[f32],
    scale_shape: &[usize],
    scale_key: &str,
    block: usize,
) -> Result<Tensor> {
    if scales.len() == 1 {
        let scale = scales
            .first()
            .ok_or_else(|| InferError::Shape(format!("scale FP8 {scale_key} vide")))?;
        return Ok(tensor.map(|value| value * *scale));
    }
    let (rows, cols) = tensor.as_matrix()?;
    let [scale_rows, scale_cols] = scale_shape else {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} attendu rang 2, reçu {scale_shape:?}"
        )));
    };
    let expected_rows = rows.div_ceil(block);
    let expected_cols = cols.div_ceil(block);
    if *scale_rows != expected_rows || *scale_cols != expected_cols {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} attendu [{expected_rows},{expected_cols}], reçu {scale_shape:?}"
        )));
    }
    if scales.len() != scale_rows * scale_cols {
        return Err(InferError::Shape(format!(
            "scale FP8 {scale_key} shape={scale_shape:?}, éléments={}",
            scales.len()
        )));
    }
    let mut out = tensor.data().to_vec();
    for row in 0..rows {
        let scale_row = row / block;
        for col in 0..cols {
            let scale_col = col / block;
            out[row * cols + col] *= scales[scale_row * scale_cols + scale_col];
        }
    }
    Tensor::from_vec(vec![rows, cols], out)
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    std::fs::read(path).map_err(|source| InferError::Io {
        path: PathBuf::from(path),
        source,
    })
}

fn bytes_to_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(4);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} F32 avec {} octets non multiple de 4",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in chunks {
        let arr = <[u8; 4]>::try_from(chunk)
            .map_err(|_| InferError::Shape(format!("chunk F32 invalide pour {name}")))?;
        out.push(f32::from_le_bytes(arr));
    }
    Ok(out)
}

fn bytes_to_bf16_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} BF16 avec {} octets non multiple de 2",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in chunks {
        let arr = <[u8; 2]>::try_from(chunk)
            .map_err(|_| InferError::Shape(format!("chunk BF16 invalide pour {name}")))?;
        let bits = u16::from_le_bytes(arr);
        out.push(f32::from_bits(u32::from(bits) << 16));
    }
    Ok(out)
}

fn bytes_to_f16_f32(bytes: &[u8], name: &str) -> Result<Vec<f32>> {
    let chunks = bytes.chunks_exact(2);
    if !chunks.remainder().is_empty() {
        return Err(InferError::Shape(format!(
            "tensor {name} F16 avec {} octets non multiple de 2",
            bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 2);
    for chunk in chunks {
        let arr = <[u8; 2]>::try_from(chunk)
            .map_err(|_| InferError::Shape(format!("chunk F16 invalide pour {name}")))?;
        out.push(f16_bits_to_f32(u16::from_le_bytes(arr)));
    }
    Ok(out)
}

fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits & 0x8000) << 16;
    let exponent = (bits >> 10) & 0x001f;
    let mantissa = bits & 0x03ff;

    let f32_bits = match (exponent, mantissa) {
        (0, 0) => sign,
        (0, _) => {
            let mut mant = u32::from(mantissa);
            let mut exp = -14_i32;
            while (mant & 0x0400) == 0 {
                mant <<= 1;
                exp -= 1;
            }
            mant &= 0x03ff;
            let exp_bits = ((exp + 127) as u32) << 23;
            sign | exp_bits | (mant << 13)
        }
        (0x1f, _) => sign | 0x7f80_0000 | (u32::from(mantissa) << 13),
        _ => {
            let exp_bits = (u32::from(exponent) + 112) << 23;
            sign | exp_bits | (u32::from(mantissa) << 13)
        }
    };
    f32::from_bits(f32_bits)
}

fn bytes_to_f8_e4m3_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(bytes.iter().copied().map(f8_e4m3_to_f32).collect())
}

fn bytes_to_f8_e5m2_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(bytes.iter().copied().map(f8_e5m2_to_f32).collect())
}

fn f8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 && mantissa == 0 {
        return sign * 0.0;
    }
    let decoded = if exponent == 0 {
        (f32::from(mantissa) / 8.0) * 2.0_f32.powi(-6)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    sign * decoded
}

fn f8_e5m2_to_f32(byte: u8) -> f32 {
    f8_to_f32(byte, 5, 2, 15)
}

fn f8_to_f32(byte: u8, exponent_bits: u8, mantissa_bits: u8, bias: i32) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent_mask = (1_u8 << exponent_bits) - 1;
    let mantissa_mask = (1_u8 << mantissa_bits) - 1;
    let exponent = (byte >> mantissa_bits) & exponent_mask;
    let mantissa = byte & mantissa_mask;

    if exponent == 0 && mantissa == 0 {
        return sign * 0.0;
    }
    if exponent == exponent_mask {
        if mantissa == 0 {
            return sign * f32::INFINITY;
        }
        return f32::NAN;
    }

    let mantissa_scale = (1_u32 << mantissa_bits) as f32;
    if exponent == 0 {
        sign * (f32::from(mantissa) / mantissa_scale) * 2.0_f32.powi(1 - bias)
    } else {
        sign * (1.0 + f32::from(mantissa) / mantissa_scale)
            * 2.0_f32.powi(i32::from(exponent) - bias)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::{serialize, View};
    use std::borrow::Cow;

    struct F32View {
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for F32View {
        fn dtype(&self) -> Dtype {
            Dtype::F32
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    #[test]
    fn loads_f32_tensor_from_safetensors() {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        let data = [1.0_f32, -2.5, 3.25]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let buffer = serialize(
            [(
                "linear.weight",
                F32View {
                    shape: vec![1, 3],
                    data,
                },
            )],
            None,
        )
        .expect("invariant: safetensors sérialisable");
        std::fs::write(tmp.path(), buffer).expect("invariant: écriture temporaire");

        let tensor =
            load_f32_tensor(tmp.path(), "linear.weight").expect("invariant: tensor F32 chargeable");
        assert_eq!(tensor.shape(), &[1, 3]);
        assert_eq!(tensor.data(), &[1.0, -2.5, 3.25]);
    }

    #[test]
    fn loads_bf16_and_f16_as_dense_f32() {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        let bf16_data = [1.0_f32, -2.0]
            .into_iter()
            .flat_map(|value| {
                let bits = (value.to_bits() >> 16) as u16;
                bits.to_le_bytes()
            })
            .collect::<Vec<_>>();
        let f16_data = [0x3c00_u16, 0xc000_u16]
            .into_iter()
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        let buffer = serialize(
            [
                (
                    "bf16.weight",
                    RawView {
                        dtype: Dtype::BF16,
                        shape: vec![1, 2],
                        data: bf16_data,
                    },
                ),
                (
                    "f16.weight",
                    RawView {
                        dtype: Dtype::F16,
                        shape: vec![1, 2],
                        data: f16_data,
                    },
                ),
            ],
            None,
        )
        .expect("invariant: safetensors sérialisable");
        std::fs::write(tmp.path(), buffer).expect("invariant: écriture temporaire");

        let bf16 =
            load_float_tensor(tmp.path(), "bf16.weight").expect("invariant: BF16 chargeable");
        let f16 = load_float_tensor(tmp.path(), "f16.weight").expect("invariant: F16 chargeable");
        assert_eq!(bf16.data(), &[1.0, -2.0]);
        assert_eq!(f16.data(), &[1.0, -2.0]);
    }

    #[test]
    fn loads_fp8_as_dense_f32() {
        let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
        let buffer = serialize(
            [
                (
                    "e4m3.weight",
                    RawView {
                        dtype: Dtype::F8_E4M3,
                        shape: vec![1, 3],
                        data: vec![0x38, 0x40, 0xb8],
                    },
                ),
                (
                    "e5m2.weight",
                    RawView {
                        dtype: Dtype::F8_E5M2,
                        shape: vec![1, 3],
                        data: vec![0x3c, 0x40, 0xbc],
                    },
                ),
            ],
            None,
        )
        .expect("invariant: safetensors sérialisable");
        std::fs::write(tmp.path(), buffer).expect("invariant: écriture temporaire");

        let e4m3 =
            load_float_tensor(tmp.path(), "e4m3.weight").expect("invariant: E4M3 chargeable");
        let e5m2 =
            load_float_tensor(tmp.path(), "e5m2.weight").expect("invariant: E5M2 chargeable");

        assert_eq!(e4m3.data(), &[1.0, 2.0, -1.0]);
        assert_eq!(e5m2.data(), &[1.0, 2.0, -1.0]);
    }

    struct RawView {
        dtype: Dtype,
        shape: Vec<usize>,
        data: Vec<u8>,
    }

    impl View for RawView {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }
}
