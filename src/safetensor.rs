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
                tensor = apply_fp8_scales(
                    tensor,
                    &scales,
                    scale_view.shape(),
                    &scale_key,
                    FP8_SCALE_BLOCK,
                )?;
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
    let shape = tensor.shape().to_vec();
    let mut data = tensor.into_data();
    apply_fp8_block_scales_in_place(&mut data, &shape, scales, scale_shape, scale_key, block)?;
    Tensor::from_vec(shape, data)
}

/// Taille de bloc des échelles FP8 « fine-grained » (blocs `128×128`).
pub(crate) const FP8_SCALE_BLOCK: usize = 128;

/// Contrôle qu'une échelle FP8 par blocs vaut exactement
/// `[ceil(rows/block), ceil(cols/block)]` pour un poids `[rows, cols]`, et
/// renvoie `(rows, cols)`.
///
/// # Errors
///
/// Renvoie une erreur si le poids n'est pas une matrice, si `block` est nul ou
/// si la forme de l'échelle diffère.
pub(crate) fn check_fp8_block_scale_shape(
    shape: &[usize],
    scale_shape: &[usize],
    scale_key: &str,
    block: usize,
) -> Result<(usize, usize)> {
    let [rows, cols] = shape else {
        return Err(InferError::Dimension(format!(
            "poids FP8 à blocs attendu rang 2, reçu {shape:?} ({scale_key})"
        )));
    };
    if block == 0 {
        return Err(InferError::Config(format!(
            "scale FP8 {scale_key}: bloc nul"
        )));
    }
    let expected = [rows.div_ceil(block), cols.div_ceil(block)];
    if scale_shape != expected {
        return Err(InferError::Dimension(format!(
            "scale FP8 {scale_key} attendu {expected:?}, reçu {scale_shape:?}"
        )));
    }
    Ok((*rows, *cols))
}

/// Applique en place les échelles FP8 d'un poids déjà décodé.
///
/// Une échelle unique (`[1]` ou `[]`) multiplie tout le tenseur ; sinon le poids
/// est une matrice `[rows, cols]` et `scale_shape` vaut exactement
/// `[ceil(rows/block), ceil(cols/block)]` (blocs partiels en bord inclus). Le
/// fichier stocke `weight_scale_inv` déjà inversée : `w = e4m3 · scale`.
///
/// # Errors
///
/// Renvoie une erreur si la forme de l'échelle ne correspond pas au poids ou si
/// une échelle n'est pas finie.
pub(crate) fn apply_fp8_block_scales_in_place(
    values: &mut [f32],
    shape: &[usize],
    scales: &[f32],
    scale_shape: &[usize],
    scale_key: &str,
    block: usize,
) -> Result<()> {
    if let Some(bad) = scales.iter().find(|scale| !scale.is_finite()) {
        return Err(InferError::Config(format!(
            "scale FP8 {scale_key} non finie: {bad}"
        )));
    }
    if scales.len() == 1 {
        let scale = scales[0];
        values.iter_mut().for_each(|value| *value *= scale);
        return Ok(());
    }
    let (rows, cols) = check_fp8_block_scale_shape(shape, scale_shape, scale_key, block)?;
    let scale_cols = cols.div_ceil(block);
    if scales.len() != rows.div_ceil(block) * scale_cols || values.len() != rows * cols {
        return Err(InferError::Shape(format!(
            "scale FP8 {scale_key} shape={scale_shape:?}, éléments={}, poids {}",
            scales.len(),
            values.len()
        )));
    }
    for (row, out_row) in values.chunks_exact_mut(cols).enumerate() {
        let scale_row = &scales[(row / block) * scale_cols..(row / block + 1) * scale_cols];
        for (col, value) in out_row.iter_mut().enumerate() {
            *value *= scale_row[col / block];
        }
    }
    Ok(())
}

/// Lit exactement une plage du fichier, sans dépendre du curseur partagé.
/// Sous Windows, `seek_read` peut renvoyer une lecture partielle.
pub(crate) fn pread_exact(
    file: &std::fs::File,
    buf: &mut [u8],
    offset: u64,
) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        read_exact_at_with(buf, offset, |buf, offset| file.seek_read(buf, offset))
    }
}

#[cfg(any(windows, test))]
fn read_exact_at_with(
    mut buf: &mut [u8],
    mut offset: u64,
    mut read: impl FnMut(&mut [u8], u64) -> std::io::Result<usize>,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    while !buf.is_empty() {
        match read(buf, offset) {
            Ok(0) => return Err(ErrorKind::UnexpectedEof.into()),
            Ok(n) => {
                buf = &mut buf[n..];
                if !buf.is_empty() {
                    offset = offset.checked_add(n as u64).ok_or_else(|| {
                        Error::new(ErrorKind::InvalidInput, "offset de lecture hors plage")
                    })?;
                }
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
}

/// Écrit intégralement une plage, y compris après une écriture partielle.
#[cfg(feature = "devtools")]
pub(crate) fn pwrite_all(file: &std::fs::File, buf: &[u8], offset: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.write_all_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        write_all_at_with(buf, offset, |buf, offset| file.seek_write(buf, offset))
    }
}

#[cfg(all(feature = "devtools", any(windows, test)))]
fn write_all_at_with(
    mut buf: &[u8],
    mut offset: u64,
    mut write: impl FnMut(&[u8], u64) -> std::io::Result<usize>,
) -> std::io::Result<()> {
    use std::io::{Error, ErrorKind};
    while !buf.is_empty() {
        match write(buf, offset) {
            Ok(0) => return Err(ErrorKind::WriteZero.into()),
            Ok(n) => {
                buf = &buf[n..];
                if !buf.is_empty() {
                    offset = offset.checked_add(n as u64).ok_or_else(|| {
                        Error::new(ErrorKind::InvalidInput, "offset d'écriture hors plage")
                    })?;
                }
            }
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        }
    }
    Ok(())
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
    let mut out = vec![0.0_f32; bytes.len()];
    decode_f8_e4m3_into(bytes, &mut out, "e4m3")?;
    Ok(out)
}

fn bytes_to_f8_e5m2_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    Ok(bytes.iter().copied().map(f8_e5m2_to_f32).collect())
}

/// Décode des octets E4M3FN dans `out`.
///
/// # Errors
///
/// Renvoie une erreur si les longueurs diffèrent ou si un octet code NaN
/// (`0x7f`/`0xff` : E4M3FN n'a pas d'infini, ces deux codes sont ses seuls NaN).
pub(crate) fn decode_f8_e4m3_into(bytes: &[u8], out: &mut [f32], name: &str) -> Result<()> {
    if bytes.len() != out.len() {
        return Err(InferError::Shape(format!(
            "décodage E4M3 {name}: {} octets pour {} valeurs",
            bytes.len(),
            out.len()
        )));
    }
    for (index, (slot, &byte)) in out.iter_mut().zip(bytes).enumerate() {
        *slot = f8_e4m3_to_f32(byte).ok_or_else(|| {
            InferError::Config(format!(
                "E4M3 {name}: NaN (0x{byte:02x}) à l'indice {index}"
            ))
        })?;
    }
    Ok(())
}

/// Valeur finie d'un octet E4M3FN, `None` pour ses deux codes NaN.
pub(crate) fn f8_e4m3_to_f32(byte: u8) -> Option<f32> {
    if byte & 0x7f == 0x7f {
        return None;
    }
    let sign = if byte & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (byte >> 3) & 0x0f;
    let mantissa = byte & 0x07;
    if exponent == 0 && mantissa == 0 {
        return Some(sign * 0.0);
    }
    let decoded = if exponent == 0 {
        (f32::from(mantissa) / 8.0) * 2.0_f32.powi(-6)
    } else {
        (1.0 + f32::from(mantissa) / 8.0) * 2.0_f32.powi(i32::from(exponent) - 7)
    };
    Some(sign * decoded)
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

    #[test]
    fn positioned_reads_cover_ranges_empty_buffers_and_eof() -> std::io::Result<()> {
        use std::io::Write;
        let mut file = tempfile::tempfile()?;
        file.write_all(b"abcdef")?;
        let mut bytes = [0; 3];
        pread_exact(&file, &mut bytes, 2)?;
        assert_eq!(&bytes, b"cde");
        pread_exact(&file, &mut [], u64::MAX)?;
        assert_eq!(
            pread_exact(&file, &mut bytes, 4).unwrap_err().kind(),
            std::io::ErrorKind::UnexpectedEof
        );
        // Les lecteurs parallèles partagent le fichier mais pas leur offset.
        std::thread::scope(|scope| {
            let file = &file;
            let readers: Vec<_> = (0..6)
                .map(|offset| {
                    scope.spawn(move || {
                        let mut byte = [0];
                        pread_exact(file, &mut byte, offset).unwrap();
                        assert_eq!(byte[0], b'a' + offset as u8);
                    })
                })
                .collect();
            for reader in readers {
                reader.join().unwrap();
            }
        });
        Ok(())
    }

    #[test]
    fn positioned_read_loop_retries_and_advances_after_short_reads() -> std::io::Result<()> {
        use std::io::ErrorKind;
        let mut bytes = [0; 3];
        let mut calls = 0;
        read_exact_at_with(&mut bytes, 5, |buf, offset| {
            calls += 1;
            if calls == 1 {
                return Err(ErrorKind::Interrupted.into());
            }
            buf[0] = offset as u8;
            Ok(1)
        })?;
        assert_eq!(bytes, [5, 6, 7]);
        assert_eq!(calls, 4);
        assert_eq!(
            read_exact_at_with(&mut bytes, 0, |_, _| Ok(0))
                .unwrap_err()
                .kind(),
            ErrorKind::UnexpectedEof
        );
        assert_eq!(
            read_exact_at_with(&mut bytes, u64::MAX, |_, _| Ok(1))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            read_exact_at_with(
                &mut bytes,
                0,
                |_, _| Err(ErrorKind::PermissionDenied.into())
            )
            .unwrap_err()
            .kind(),
            ErrorKind::PermissionDenied
        );
        read_exact_at_with(&mut [], 0, |_, _| {
            panic!("invariant: aucune lecture pour une plage vide")
        })?;
        Ok(())
    }

    #[cfg(feature = "devtools")]
    #[test]
    fn positioned_writes_handle_partial_writes_and_preserve_other_bytes() -> std::io::Result<()> {
        use std::io::{ErrorKind, Write};
        let mut file = tempfile::tempfile()?;
        file.write_all(b"abcdef")?;
        pwrite_all(&file, b"XY", 2)?;
        let mut bytes = [0; 6];
        pread_exact(&file, &mut bytes, 0)?;
        assert_eq!(&bytes, b"abXYef");
        let mut written = Vec::new();
        let mut calls = 0;
        write_all_at_with(b"abc", 7, |buf, offset| {
            calls += 1;
            if calls == 1 {
                return Err(ErrorKind::Interrupted.into());
            }
            written.push((offset, buf[0]));
            Ok(1)
        })?;
        assert_eq!(written, [(7, b'a'), (8, b'b'), (9, b'c')]);
        assert_eq!(
            write_all_at_with(b"a", 0, |_, _| Ok(0)).unwrap_err().kind(),
            ErrorKind::WriteZero
        );
        assert_eq!(
            write_all_at_with(b"ab", u64::MAX, |_, _| Ok(1))
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            write_all_at_with(b"a", 0, |_, _| Err(ErrorKind::PermissionDenied.into()))
                .unwrap_err()
                .kind(),
            ErrorKind::PermissionDenied
        );
        write_all_at_with(b"", 0, |_, _| {
            panic!("invariant: aucune écriture pour une plage vide")
        })?;
        Ok(())
    }

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

#[cfg(test)]
mod fp8_tests {
    use super::*;

    #[test]
    fn e4m3_decodes_all_finite_codes_and_rejects_nan() {
        let mut finite = 0;
        for byte in 0_u8..=255 {
            match f8_e4m3_to_f32(byte) {
                Some(value) => {
                    finite += 1;
                    assert!(value.is_finite() && value.abs() <= 448.0, "0x{byte:02x}");
                }
                None => assert!(byte == 0x7f || byte == 0xff, "0x{byte:02x}"),
            }
        }
        assert_eq!(finite, 254);
        assert_eq!(f8_e4m3_to_f32(0x7e), Some(448.0));
        assert_eq!(f8_e4m3_to_f32(0xfe), Some(-448.0));
        // Plus petit sous-normal : 2^-9.
        assert_eq!(f8_e4m3_to_f32(0x01), Some(2.0_f32.powi(-9)));
        assert_eq!(f8_e4m3_to_f32(0x08), Some(2.0_f32.powi(-6)));
        assert_eq!(f8_e4m3_to_f32(0x38), Some(1.0));
    }

    #[test]
    fn e4m3_nan_bytes_are_errors() {
        let mut out = [0.0_f32; 3];
        assert!(decode_f8_e4m3_into(&[0x38, 0x7f, 0x00], &mut out, "t").is_err());
        assert!(decode_f8_e4m3_into(&[0xff], &mut out[..1], "t").is_err());
        assert!(decode_f8_e4m3_into(&[0x38, 0x00], &mut out, "t").is_err());
        decode_f8_e4m3_into(&[0x38, 0xb8, 0x00], &mut out, "t").expect("invariant: codes finis");
        assert_eq!(out, [1.0, -1.0, 0.0]);
    }

    #[test]
    fn synthetic_partial_blocks_scale_edges() {
        // Synthétique : 3×3 avec blocs de 2 (toutes les dimensions réelles du
        // checkpoint sont multiples de 128, les blocs partiels n'y existent pas).
        let mut values = vec![1.0_f32; 9];
        apply_fp8_block_scales_in_place(
            &mut values,
            &[3, 3],
            &[1.0, 2.0, 3.0, 4.0],
            &[2, 2],
            "t",
            2,
        )
        .expect("invariant: forme cohérente");
        assert_eq!(values, [1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 3.0, 3.0, 4.0]);
    }

    #[test]
    fn block_scales_reject_bad_shape_and_non_finite() {
        let mut values = vec![1.0_f32; 4];
        assert!(apply_fp8_block_scales_in_place(
            &mut values,
            &[2, 2],
            &[1.0, 2.0],
            &[1, 2],
            "t",
            2
        )
        .is_err());
        assert!(
            apply_fp8_block_scales_in_place(&mut values, &[2, 2], &[f32::NAN], &[1], "t", 2)
                .is_err()
        );
        assert!(apply_fp8_block_scales_in_place(
            &mut values,
            &[2, 2],
            &[f32::INFINITY, 1.0, 1.0, 1.0],
            &[2, 2],
            "t",
            1
        )
        .is_err());
        apply_fp8_block_scales_in_place(&mut values, &[2, 2], &[0.5], &[1], "t", 128)
            .expect("invariant: échelle per-tensor");
        assert_eq!(values, [0.5; 4]);
    }
}
