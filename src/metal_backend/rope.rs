//! Tampons persistants des fréquences RoPE calculées sur l’hôte.

use super::*;

/// Le setup partage les tables entre couches ; les encodeurs résidents
/// empruntent ensuite leur poignée sans consulter ce cache.
#[derive(Debug, Default)]
pub(crate) struct RopeFrequencyCache(Mutex<HashMap<(u32, usize), Buffer>>);

impl RopeFrequencyCache {
    pub(crate) fn get(&self, device: &Device, theta: f32, frequency_dim: usize) -> Result<Buffer> {
        let mut cache = self.0.lock().map_err(|_| {
            InferError::Config("cache fréquences RoPE Metal empoisonné".to_string())
        })?;
        let key = (theta.to_bits(), frequency_dim);
        if let Some(buffer) = cache.get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = create_rope_frequencies(device, theta, frequency_dim)?;
        cache.insert(key, buffer.clone());
        Ok(buffer)
    }
}

/// Construit la table hors encodage ; les tests et le setup gardent la poignée.
pub(crate) fn create_rope_frequencies(
    device: &Device,
    theta: f32,
    frequency_dim: usize,
) -> Result<Buffer> {
    // Garde la borne des encodeurs avant de dimensionner l’allocation hôte.
    u32::try_from(frequency_dim)
        .map_err(|_| InferError::Dimension("rope frequency_dim hors u32".to_string()))?;
    let frequencies = crate::rope::inv_freq(frequency_dim, theta);
    // NoPE garde la RMSNorm fusionnée : Metal exige une liaison valide,
    // mais aucune fréquence n’est lue lorsque rope_dims vaut zéro.
    let buffer = if frequencies.is_empty() {
        device.new_buffer(4, MTLResourceOptions::StorageModeShared)
    } else {
        device.new_buffer_with_data(
            frequencies.as_ptr().cast::<c_void>(),
            std::mem::size_of_val(frequencies.as_slice()) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    };
    Ok(buffer)
}

impl MetalExecutor {
    pub(crate) fn rope_frequencies(&self, theta: f32, frequency_dim: usize) -> Result<Buffer> {
        self.rope_frequencies
            .get(&self.device, theta, frequency_dim)
    }
}

/// L'écart rotate-half est le nombre de fréquences de la table exacte créée
/// ci-dessus : head_dim/2 pour Gemma proportional, rope_dims/2 pour Qwen.
/// Le tampon factice NoPE n'est jamais lu (aucune paire active).
pub(crate) fn rope_half_offset(frequencies: &BufferRef) -> f32 {
    (frequencies.length() / std::mem::size_of::<f32>() as u64) as f32
}
