//! Poids CPU du codec relus seulement si le chemin de référence les utilise.

use crate::tts::SafetensorPayload;
use crate::{InferError, Result, Tensor};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::sync::OnceLock;

#[derive(Debug)]
struct Weight {
    value: OnceLock<Tensor>,
    /// Vérifie les bits relus : un snapshot remplacé ou modifié doit échouer.
    digest: Option<[u8; 32]>,
}

#[derive(Debug)]
pub(super) struct CodecWeights {
    entries: HashMap<String, Weight>,
    payload: SafetensorPayload,
}

fn digest(tensor: &Tensor) -> [u8; 32] {
    Sha256::digest(bytemuck::cast_slice(tensor.data())).into()
}

impl CodecWeights {
    pub(super) fn new(
        weights: HashMap<String, Tensor>,
        payload: &SafetensorPayload,
        discard: impl Fn(&str) -> bool,
    ) -> Self {
        let entries = weights
            .into_iter()
            .map(|(name, tensor)| {
                let weight = if discard(&name) && payload.contains(&name) {
                    Weight {
                        value: OnceLock::new(),
                        digest: Some(digest(&tensor)),
                    }
                } else {
                    Weight {
                        value: OnceLock::from(tensor),
                        digest: None,
                    }
                };
                (name, weight)
            })
            .collect();
        Self {
            entries,
            payload: payload.clone(),
        }
    }

    pub(super) fn get(&self, key: &str) -> Result<Option<&Tensor>> {
        let Some(weight) = self.entries.get(key) else {
            return Ok(None);
        };
        if let Some(value) = weight.value.get() {
            return Ok(Some(value));
        }
        let value = self.payload.read_dense_tensor(key)?;
        if weight.digest != Some(digest(&value)) {
            return Err(InferError::Config(format!(
                "poids codec modifié depuis l'upload: {key}"
            )));
        }
        // Deux lecteurs peuvent relire simultanément le même poids immuable.
        // Le premier gagne ; la copie perdante est libérée immédiatement.
        let _ = weight.value.set(value);
        weight
            .value
            .get()
            .map(Some)
            .ok_or_else(|| InferError::Config(format!("poids codec non initialisé: {key}")))
    }

    pub(super) fn retained_bytes(&self) -> usize {
        self.entries
            .values()
            .filter_map(|weight| weight.value.get())
            .map(Tensor::capacity_bytes)
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::TensorView;

    fn write(path: &std::path::Path, values: &[f32]) -> Result<()> {
        let view = TensorView::new(
            safetensors::Dtype::F32,
            vec![values.len()],
            bytemuck::cast_slice(values),
        )
        .map_err(|e| InferError::Config(e.to_string()))?;
        let bytes = safetensors::serialize([("weight", view)], None)
            .map_err(|e| InferError::Config(e.to_string()))?;
        std::fs::write(path, bytes).map_err(|source| InferError::Io {
            path: path.into(),
            source,
        })
    }

    #[test]
    fn lazy_weights_survive_current_directory_change(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        const CHILD: &str = "SARAGOSSA_TEST_TTS_CWD_CHILD";
        if std::env::var_os(CHILD).is_none() {
            // Un seul test dans un processus isolé : aucun autre lecteur de
            // chemin relatif ne peut subir notre changement de cwd global.
            let output = std::process::Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "tts_codec::weights::tests::lazy_weights_survive_current_directory_change",
                    "--test-threads=1",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .output()?;
            assert!(
                output.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            return Ok(());
        }

        struct RestoreCwd(std::path::PathBuf);
        impl Drop for RestoreCwd {
            fn drop(&mut self) {
                std::env::set_current_dir(&self.0)
                    .expect("invariant: le répertoire initial existe encore");
            }
        }

        let dir = tempfile::tempdir()?;
        let other = tempfile::tempdir()?;
        // Déclarée après les répertoires temporaires, la garde restaure le cwd
        // avant leur suppression, y compris en cas d'erreur ou de panique.
        let restore = RestoreCwd(std::env::current_dir()?);
        let values = [0.0_f32, -0.0, 1.25, f32::from_bits(0x7fc00001)];
        write(&dir.path().join("codec.safetensors"), &values)?;
        std::env::set_current_dir(dir.path())?;
        let payload = SafetensorPayload::open(std::path::Path::new("codec.safetensors"))?;
        let lazy = CodecWeights::new(
            HashMap::from([("weight".into(), payload.read_dense_tensor("weight")?)]),
            &payload,
            |_| true,
        );
        assert_eq!(lazy.retained_bytes(), 0);
        std::env::set_current_dir(other.path())?;
        let loaded = lazy.get("weight")?.expect("invariant: poids présent");
        assert_eq!(
            loaded
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        let original = restore.0.clone();
        drop(restore);
        assert_eq!(std::env::current_dir()?, original);
        Ok(())
    }

    #[test]
    fn missing_payload_returns_typed_io_error(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("missing.safetensors");
        let error = SafetensorPayload::open(&path).expect_err("invariant: fichier absent");
        assert!(matches!(
            error,
            InferError::Io { path: failed, source }
                if failed == path && source.kind() == std::io::ErrorKind::NotFound
        ));
        Ok(())
    }

    #[test]
    fn lazy_weights_preserve_bits_and_reject_replaced_payloads() -> Result<()> {
        let dir = tempfile::tempdir().map_err(|source| InferError::Io {
            path: "tempdir".into(),
            source,
        })?;
        let path = dir.path().join("codec.safetensors");
        let values = [0.0_f32, -0.0, 1.25, f32::from_bits(0x7fc00001)];
        write(&path, &values)?;
        let payload = SafetensorPayload::open(&path)?;
        let make = || -> Result<CodecWeights> {
            Ok(CodecWeights::new(
                HashMap::from([("weight".into(), payload.read_dense_tensor("weight")?)]),
                &payload,
                |_| true,
            ))
        };
        let lazy = make()?;
        let changed = make()?;
        assert_eq!(lazy.retained_bytes(), 0);
        let loaded = lazy.get("weight")?.expect("invariant: poids présent");
        assert_eq!(
            loaded
                .data()
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert!(lazy.retained_bytes() >= size_of_val(&values));
        assert!(lazy.get("missing")?.is_none());
        write(&path, &[2.0; 4])?;
        assert!(changed.get("weight").is_err());
        assert_eq!(
            lazy.get("weight")?
                .expect("invariant: poids mémorisé")
                .data()
                .as_ptr(),
            loaded.data().as_ptr()
        );
        Ok(())
    }
}
