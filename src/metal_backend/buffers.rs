//! Allocation, cache et résolution des buffers Metal.

use super::*;

impl MetalExecutor {
    pub(crate) fn resolve_embedding_weight_buffers(
        &self,
        embedding: &EmbeddingWeight,
    ) -> Result<MetalEmbeddingWeightBuffers> {
        match embedding {
            EmbeddingWeight::Dense(table) => {
                let (vocab, dim) = table.as_matrix()?;
                Ok(MetalEmbeddingWeightBuffers::Dense {
                    table: self.cached_buffer_from_f32(table.data(), "resident_embed_dense")?,
                    vocab,
                    dim,
                })
            }
            EmbeddingWeight::AffineQuantized(weight) => {
                let [vocab, dim] = weight.shape() else {
                    return Err(InferError::Dimension(format!(
                        "embedding quantifié attendu rang 2, reçu {:?}",
                        weight.shape()
                    )));
                };
                let [packed_rows, packed_cols] = weight.packed_shape() else {
                    return Err(InferError::Dimension(format!(
                        "embedding packed_shape attendu rang 2, reçu {:?}",
                        weight.packed_shape()
                    )));
                };
                if *packed_rows != *vocab {
                    return Err(InferError::Dimension(format!(
                        "embedding packed_rows={packed_rows} incompatible avec vocab={vocab}"
                    )));
                }
                let groups = dim.checked_div(weight.group_size()).ok_or_else(|| {
                    InferError::Metal("group_size embedding quantifié nul".to_string())
                })?;
                Ok(MetalEmbeddingWeightBuffers::AffineQuantized {
                    packed: self
                        .cached_buffer_from_u32(weight.packed_data(), "resident_embed_packed")?,
                    scales: self.cached_buffer_from_f32_as_bf16(
                        weight.scales().data(),
                        "resident_embed_scales",
                    )?,
                    biases: self.cached_buffer_from_f32_as_bf16(
                        weight.biases().data(),
                        "resident_embed_biases",
                    )?,
                    vocab: *vocab,
                    dim: *dim,
                    packed_cols: *packed_cols,
                    group_size: weight.group_size(),
                    bits: weight.bits(),
                    groups,
                })
            }
        }
    }

    pub(crate) fn resolve_linear_attn_resident_weights(
        &self,
        weights: LinearAttnResidentWeights<'_>,
    ) -> Result<MetalLinearAttnResidentWeights> {
        Ok(MetalLinearAttnResidentWeights {
            in_proj: self.resolve_concat_linear_weight_buffers(
                &[
                    weights.in_proj_qkv.weight(),
                    weights.in_proj_z.weight(),
                    weights.in_proj_b.weight(),
                    weights.in_proj_a.weight(),
                ],
                "resident_la_in_proj_concat",
            )?,
            out_proj: self
                .resolve_linear_weight_buffers(weights.out_proj.weight(), "resident_la_out")?,
            conv_weight: self
                .cached_buffer_from_f32(weights.conv_weight.data(), "resident_la_conv_weight")?,
            a_log: self.cached_buffer_from_f32(weights.a_log, "resident_la_a_log")?,
            dt_bias: self.cached_buffer_from_f32(weights.dt_bias, "resident_la_dt_bias")?,
            norm_weight: self.cached_buffer_from_f32(weights.norm_weight, "resident_la_norm")?,
        })
    }

    pub(crate) fn resolve_linear_attn_resident_dense_weights(
        &self,
        weights: LinearAttnResidentWeights<'_>,
    ) -> Result<MetalLinearAttnResidentDenseWeights> {
        let full = match self.resolve_linear_attn_resident_weights(weights) {
            Ok(weights) => Some(weights),
            Err(InferError::Dimension(_)) => None,
            Err(error) => return Err(error),
        };
        Ok(MetalLinearAttnResidentDenseWeights {
            full,
            qkv_z: self.resolve_linear_attn_pair_weights(
                weights.in_proj_qkv,
                weights.in_proj_z,
                "resident_la_in_proj_qkv_z",
                "resident_la_in_proj_qkv",
                "resident_la_in_proj_z",
            )?,
            beta_gate: self.resolve_linear_attn_pair_weights(
                weights.in_proj_b,
                weights.in_proj_a,
                "resident_la_in_proj_beta_gate",
                "resident_la_in_proj_b",
                "resident_la_in_proj_a",
            )?,
            out_proj: self
                .resolve_linear_weight_buffers(weights.out_proj.weight(), "resident_la_out")?,
            conv_weight: self
                .cached_buffer_from_f32(weights.conv_weight.data(), "resident_la_conv_weight")?,
            a_log: self.cached_buffer_from_f32(weights.a_log, "resident_la_a_log")?,
            dt_bias: self.cached_buffer_from_f32(weights.dt_bias, "resident_la_dt_bias")?,
            norm_weight: self.cached_buffer_from_f32(weights.norm_weight, "resident_la_norm")?,
        })
    }

    fn resolve_linear_attn_pair_weights(
        &self,
        first: &Linear,
        second: &Linear,
        concat_label: &'static str,
        first_label: &'static str,
        second_label: &'static str,
    ) -> Result<MetalLinearAttnResidentPairWeights> {
        match self
            .resolve_concat_linear_weight_buffers(&[first.weight(), second.weight()], concat_label)
        {
            Ok(weights) => Ok(MetalLinearAttnResidentPairWeights::Concat(weights)),
            Err(InferError::Dimension(_)) => Ok(MetalLinearAttnResidentPairWeights::Split {
                first: self.resolve_linear_weight_buffers(first.weight(), first_label)?,
                second: self.resolve_linear_weight_buffers(second.weight(), second_label)?,
            }),
            Err(error) => Err(error),
        }
    }

    pub(crate) fn resolve_moe_shared_weights(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<MetalMoeSharedWeights> {
        ensure_biasless(router, "router")?;
        ensure_biasless(shared_gate, "shared_gate")?;
        let (shared_gate_proj, shared_up_proj, shared_down_proj) = shared_expert.projections();
        ensure_biasless(shared_gate_proj, "shared_gate_proj")?;
        ensure_biasless(shared_up_proj, "shared_up_proj")?;
        ensure_biasless(shared_down_proj, "shared_down_proj")?;
        Ok(MetalMoeSharedWeights {
            router: self.resolve_linear_weight_buffers(router.weight(), "resident_moe_router")?,
            stacked: self.stacked_moe_buffers(experts)?,
            shared_gate: self
                .resolve_linear_weight_buffers(shared_gate.weight(), "resident_shared_gate")?,
            shared_gate_proj: self.resolve_linear_weight_buffers(
                shared_gate_proj.weight(),
                "resident_shared_gate_proj",
            )?,
            shared_up_proj: self.resolve_linear_weight_buffers(
                shared_up_proj.weight(),
                "resident_shared_up_proj",
            )?,
            shared_down_proj: self.resolve_linear_weight_buffers(
                shared_down_proj.weight(),
                "resident_shared_down_proj",
            )?,
        })
    }

    pub(crate) fn resolve_moe_routed_weights(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
    ) -> Result<MetalMoeRoutedWeights> {
        ensure_biasless(router, "router")?;
        Ok(MetalMoeRoutedWeights {
            router: self.resolve_linear_weight_buffers(router.weight(), "resident_moe_router")?,
            stacked: self.stacked_moe_buffers(experts)?,
        })
    }

    pub(crate) fn linear_weight_out_dim(&self, weight: &MetalLinearWeightBuffers) -> usize {
        match weight {
            MetalLinearWeightBuffers::Dense { out_dim, .. }
            | MetalLinearWeightBuffers::AffineQuantized { out_dim, .. } => *out_dim,
        }
    }

    pub(crate) fn linear_weight_in_dim(&self, weight: &MetalLinearWeightBuffers) -> usize {
        match weight {
            MetalLinearWeightBuffers::Dense { in_dim, .. }
            | MetalLinearWeightBuffers::AffineQuantized { in_dim, .. } => *in_dim,
        }
    }

    pub(super) fn buffer_from_f32(
        &self,
        data: &[f32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.buffer_from_slice(data, label)
    }

    pub(super) fn buffer_from_u32(
        &self,
        data: &[u32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.buffer_from_slice(data, label)
    }

    pub(super) fn upload_f32_buffer(
        &self,
        data: &[f32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        let buffer = self.scratch_buffer(data.len(), MetalBufferElement::F32, label)?;
        write_f32_buffer(&buffer, data)?;
        Ok(buffer)
    }

    pub(super) fn upload_u32_buffer(
        &self,
        data: &[u32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        let buffer = self.scratch_buffer(data.len(), MetalBufferElement::U32, label)?;
        write_u32_buffer(&buffer, data)?;
        Ok(buffer)
    }

    /// Renvoie un buffer Metal résident pour `data`, mémoïsé par adresse du
    /// pointeur (les poids/normes ont une adresse stable entre tokens → un seul
    /// upload). Exposé `pub(crate)` pour bufferiser les tenseurs de norme du
    /// decode résident (1c).
    pub(crate) fn cached_buffer_from_f32(
        &self,
        data: &[f32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.cached_buffer(
            data.as_ptr().addr(),
            data.len(),
            MetalBufferElement::F32,
            label,
            || self.buffer_from_f32(data, label),
        )
    }

    pub(super) fn cached_buffer_from_u32(
        &self,
        data: &[u32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.cached_buffer(
            data.as_ptr().addr(),
            data.len(),
            MetalBufferElement::U32,
            label,
            || self.buffer_from_u32(data, label),
        )
    }

    /// Renvoie un buffer Metal résident bf16 (arrondi RNE depuis f32), mémoïsé par
    /// l'adresse du pointeur f32 source → un seul upload (au chargement, pas par token).
    ///
    /// Utilisé pour les scales/biases quantifiés : les kernels qmv/swiglu/gather/argmax
    /// les lisent en `bfloat` (accumulation `float` inchangée), divisant par deux leur
    /// trafic mémoire (≈ 8 % du trafic poids decode) et alignant les numériques sur mlx.
    pub(crate) fn cached_buffer_from_f32_as_bf16(
        &self,
        data: &[f32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.cached_buffer(
            data.as_ptr().addr(),
            data.len(),
            MetalBufferElement::Bf16,
            label,
            || self.buffer_from_f32_as_bf16(data, label),
        )
    }

    /// Upload bf16 (arrondi RNE) non mémoïsé — pour les buffers concaténés
    /// (qkv_split, experts MoE empilés) construits à la volée.
    pub(super) fn buffer_from_f32_as_bf16(
        &self,
        data: &[f32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.buffer_from_slice(&f32_slice_to_bf16(data), label)
    }

    pub(super) fn cached_buffer(
        &self,
        ptr: usize,
        len: usize,
        element: MetalBufferElement,
        label: &'static str,
        create: impl FnOnce() -> Result<metal::Buffer>,
    ) -> Result<metal::Buffer> {
        let key = MetalBufferKey { ptr, len, element };
        let mut buffers = self
            .weight_buffers
            .lock()
            .map_err(|_| InferError::Metal(format!("cache buffer Metal empoisonné: {label}")))?;
        if let Some(buffer) = buffers.get(&key) {
            return Ok(buffer.clone());
        }
        let buffer = create()?;
        buffers.insert(key, buffer.clone());
        Ok(buffer)
    }

    pub(super) fn buffer_from_slice<T>(
        &self,
        data: &[T],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if data.is_empty() {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        let bytes = byte_len_usize::<T>(data.len())?;
        Ok(self.device.new_buffer_with_data(
            data.as_ptr().cast::<c_void>(),
            checked_nsuint(bytes, label)?,
            MTLResourceOptions::StorageModeShared,
        ))
    }

    pub(super) fn new_f32_buffer(&self, len: usize, label: &'static str) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer(len, MetalBufferElement::F32, label)
    }

    pub(super) fn new_u32_buffer(&self, len: usize, label: &'static str) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer(len, MetalBufferElement::U32, label)
    }

    pub(super) fn uncached_f32_buffer(
        &self,
        len: usize,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        Ok(self
            .device
            .new_buffer(byte_len::<f32>(len)?, MTLResourceOptions::StorageModeShared))
    }

    pub(super) fn private_f32_buffer(
        &self,
        len: usize,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer_with_options(
            len,
            MetalBufferElement::F32,
            label,
            scratch_resource_options(),
        )
    }

    pub(super) fn private_u32_buffer(
        &self,
        len: usize,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer_with_options(
            len,
            MetalBufferElement::U32,
            label,
            scratch_resource_options(),
        )
    }

    pub(super) fn scratch_buffer(
        &self,
        len: usize,
        element: MetalBufferElement,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.scratch_buffer_with_options(len, element, label, MTLResourceOptions::StorageModeShared)
    }

    pub(super) fn scratch_buffer_with_options(
        &self,
        len: usize,
        element: MetalBufferElement,
        label: &'static str,
        options: MTLResourceOptions,
    ) -> Result<metal::Buffer> {
        let key = ScratchBufferKey {
            label,
            len,
            element,
        };
        let mut buffers = self
            .scratch_buffers
            .lock()
            .map_err(|_| InferError::Metal(format!("cache scratch Metal empoisonné: {label}")))?;
        if let Some(buffer) = buffers.get(&key) {
            return Ok(buffer.clone());
        }
        let bytes = match element {
            MetalBufferElement::F32 => byte_len::<f32>(len)?,
            MetalBufferElement::U32 => byte_len::<u32>(len)?,
            MetalBufferElement::Bf16 => byte_len::<u16>(len)?,
        };
        let buffer = self.device.new_buffer(bytes, options);
        buffers.insert(key, buffer.clone());
        Ok(buffer)
    }

    pub(super) fn qmv_thread_group_size(&self, pipeline: &ComputePipelineState) -> NSUInteger {
        pipeline
            .thread_execution_width()
            .min(pipeline.max_total_threads_per_threadgroup())
            .max(1)
    }

    /// Renvoie le device Metal (pour bâtir l'arène résidente du decode full-attn).
    pub(crate) fn device(&self) -> &Device {
        &self.device
    }
}

/// Convertit un slice f32 en bf16 (`u16`) par arrondi au plus proche pair (RNE),
/// identique à la troncature-haute de mantisse de mlx pour les scales/biases.
fn f32_slice_to_bf16(data: &[f32]) -> Vec<u16> {
    data.iter()
        .map(|&v| {
            let bits = v.to_bits();
            // RNE : ajoute 0x7fff + bit de poids faible conservé avant de tronquer.
            let rounding = 0x7fff + ((bits >> 16) & 1);
            ((bits + rounding) >> 16) as u16
        })
        .collect()
}
