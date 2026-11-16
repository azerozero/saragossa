//! Comptabilité physique des caches Metal pour les sondes TTS.

use super::*;

impl MetalExecutor {
    pub(crate) fn tts_memory_stats(&self) -> Vec<(&'static str, usize)> {
        let mut seen = std::collections::HashSet::new();
        let mut count = |buffer: &Buffer| {
            if seen.insert(buffer.as_ptr() as usize) {
                buffer.allocated_size() as usize
            } else {
                0
            }
        };
        let weights = self
            .weight_buffers
            .lock()
            .map(|cache| cache.values().map(&mut count).sum())
            .unwrap_or(0);
        let transposed = self
            .bf16_rhs_t_cache
            .lock()
            .map(|cache| cache.values().map(&mut count).sum())
            .unwrap_or(0);
        let concat = self
            .concat_buffers
            .lock()
            .map(|cache| {
                cache
                    .values()
                    .map(|entry| match entry {
                        ConcatCacheEntry::Incompatible => 0,
                        ConcatCacheEntry::Buffers(weight) => match weight {
                            MetalLinearWeightBuffers::Dense { rhs, rhs_bf16, .. } => {
                                count(rhs) + rhs_bf16.as_ref().map(&mut count).unwrap_or(0)
                            }
                            MetalLinearWeightBuffers::DenseBf16 { rhs, .. } => count(rhs),
                            MetalLinearWeightBuffers::AffineQuantized {
                                packed,
                                scales,
                                biases,
                                ..
                            } => count(packed) + count(scales) + count(biases),
                            MetalLinearWeightBuffers::MicroScaled(weight) => {
                                count(&weight.packed) + count(&weight.scales)
                            }
                        },
                    })
                    .sum()
            })
            .unwrap_or(0);
        let scratch = self
            .scratch_buffers
            .lock()
            .map(|cache| {
                cache
                    .entries
                    .values()
                    .map(|entry| count(&entry.buffer))
                    .sum()
            })
            .unwrap_or(0);
        vec![
            ("weight_cache", weights),
            ("transposed_cache", transposed),
            ("concat_cache", concat),
            ("backend_scratch", scratch),
        ]
    }
}
