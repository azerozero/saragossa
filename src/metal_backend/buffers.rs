//! Allocation, cache et résolution des buffers Metal.

use super::moe::MoeStackTraceContext;
use super::*;

impl MetalExecutor {
    /// Force la conservation des buffers dérivés nécessaires après libération CPU.
    pub(crate) fn prepare_cpu_weight_release(&self) {
        self.preserve_weight_caches
            .store(true, std::sync::atomic::Ordering::Release);
    }

    pub(super) fn preserving_weight_caches(&self) -> bool {
        self.preserve_weight_caches
            .load(std::sync::atomic::Ordering::Acquire)
    }

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
                    packed: self.cached_affine_packed(weight, "resident_embed_packed")?,
                    packed_offset: affine_packed_offset(weight)?,
                    scales: self.cached_affine_scales(weight, "resident_embed_scales")?,
                    scales_offset: affine_scales_offset(weight)?,
                    biases: self.cached_affine_biases(weight, "resident_embed_biases")?,
                    biases_offset: affine_biases_offset(weight)?,
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

    pub(crate) fn embed_weight_tokens(
        &self,
        embedding: &EmbeddingWeight,
        token_ids: &[usize],
        embedding_scale: f32,
        recast_bf16: bool,
    ) -> Result<Tensor> {
        let Some(&dim) = embedding.shape().get(1) else {
            return Err(InferError::Dimension(format!(
                "embedding attendu rang 2, reçu {:?}",
                embedding.shape()
            )));
        };
        if embedding.shape().len() != 2 || token_ids.is_empty() {
            return Err(InferError::Dimension(format!(
                "embedding ou tokens invalides: shape={:?}, tokens={}",
                embedding.shape(),
                token_ids.len()
            )));
        }
        let vocab = embedding.shape()[0];
        let indices = token_ids
            .iter()
            .map(|token| {
                if *token >= vocab {
                    return Err(InferError::Dimension(format!(
                        "token id {token} hors vocab {vocab}"
                    )));
                }
                u32::try_from(*token).map_err(|_| {
                    InferError::Dimension(format!("token embedding hors plage u32: {token}"))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let weights = self.resolve_embedding_weight_buffers(embedding)?;
        let indices = self.upload_u32_buffer(&indices, "embedding_indices")?;
        let output_len = checked_len(token_ids.len(), dim, "embedding Metal")?;
        let output = self.uncached_f32_buffer(output_len, "embedding_output")?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        self.encode_embeddings_from_indices_scaled(
            encoder,
            &weights,
            &indices,
            &output,
            dim,
            token_ids.len(),
            embedding_scale,
            recast_bf16,
        )?;
        encoder.end_encoding();
        commit_and_wait(command_buffer)?;
        Tensor::from_vec(
            vec![token_ids.len(), dim],
            read_f32_buffer(&output, output_len)?,
        )
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
        let full_sources = [
            weights.in_proj_qkv.weight(),
            weights.in_proj_z.weight(),
            weights.in_proj_b.weight(),
            weights.in_proj_a.weight(),
        ];
        let full = if self.concat_duplication_worth_avoiding(&full_sources) {
            None
        } else {
            match self.resolve_linear_attn_resident_weights(weights) {
                Ok(weights) => Some(weights),
                Err(InferError::Dimension(error)) => {
                    if crate::runtime_flags::trace_resident_enabled() {
                        eprintln!("linear-attn resident full concat: fallback ({error})");
                    }
                    None
                }
                Err(error) => return Err(error),
            }
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
            z_beta_gate: {
                let sources = [
                    weights.in_proj_z.weight(),
                    weights.in_proj_b.weight(),
                    weights.in_proj_a.weight(),
                ];
                if self.concat_duplication_worth_avoiding(&sources) {
                    None
                } else {
                    match self.resolve_concat_linear_weight_buffers(
                        &sources,
                        "resident_la_in_proj_z_beta_gate",
                    ) {
                        Ok(weights) => Some(weights),
                        Err(InferError::Dimension(_)) => None,
                        Err(error) => return Err(error),
                    }
                }
            },
            out_proj: self
                .resolve_linear_weight_buffers(weights.out_proj.weight(), "resident_la_out")?,
            conv_weight: self
                .cached_buffer_from_f32(weights.conv_weight.data(), "resident_la_conv_weight")?,
            a_log: self.cached_buffer_from_f32(weights.a_log, "resident_la_a_log")?,
            dt_bias: self.cached_buffer_from_f32(weights.dt_bias, "resident_la_dt_bias")?,
            norm_weight: self.cached_buffer_from_f32(weights.norm_weight, "resident_la_norm")?,
        })
    }

    pub(super) fn resolve_linear_attn_pair_weights(
        &self,
        first: &Linear,
        second: &Linear,
        concat_label: &'static str,
        first_label: &'static str,
        second_label: &'static str,
    ) -> Result<MetalLinearAttnResidentPairWeights> {
        let sources = [first.weight(), second.weight()];
        let resolved = if self.concat_would_duplicate_affine_storage(&sources) {
            MetalLinearAttnResidentPairWeights::Split {
                first: self.resolve_linear_weight_buffers(first.weight(), first_label)?,
                second: self.resolve_linear_weight_buffers(second.weight(), second_label)?,
            }
        } else {
            match self.resolve_concat_linear_weight_buffers(&sources, concat_label) {
                Ok(weights) => MetalLinearAttnResidentPairWeights::Concat(weights),
                Err(InferError::Dimension(_)) => MetalLinearAttnResidentPairWeights::Split {
                    first: self.resolve_linear_weight_buffers(first.weight(), first_label)?,
                    second: self.resolve_linear_weight_buffers(second.weight(), second_label)?,
                },
                Err(error) => return Err(error),
            }
        };
        if crate::runtime_flags::trace_resident_enabled() {
            trace_linear_attn_pair(concat_label, &resolved);
        }
        Ok(resolved)
    }

    pub(crate) fn resolve_moe_shared_weights(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<MetalMoeSharedWeights> {
        self.resolve_moe_shared_weights_with_context(
            router,
            experts,
            shared_expert,
            shared_gate,
            MoeStackTraceContext::runtime(),
        )
    }

    pub(crate) fn resolve_moe_shared_weights_for_cpu_release(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        layer_index: Option<usize>,
    ) -> Result<MetalMoeSharedWeights> {
        self.resolve_moe_shared_weights_with_context(
            router,
            experts,
            shared_expert,
            shared_gate,
            MoeStackTraceContext::cpu_release(layer_index),
        )
    }

    pub(crate) fn resolve_moe_shared_weights_for_prefill(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        layer_index: usize,
    ) -> Result<MetalMoeSharedWeights> {
        self.resolve_moe_shared_weights_with_context(
            router,
            experts,
            shared_expert,
            shared_gate,
            MoeStackTraceContext::prefill(layer_index),
        )
    }

    fn resolve_moe_shared_weights_with_context(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        trace_context: MoeStackTraceContext,
    ) -> Result<MetalMoeSharedWeights> {
        ensure_biasless(router, "router")?;
        ensure_biasless(shared_gate, "shared_gate")?;
        let (shared_gate_proj, shared_up_proj, shared_down_proj) = shared_expert.projections();
        ensure_biasless(shared_gate_proj, "shared_gate_proj")?;
        ensure_biasless(shared_up_proj, "shared_up_proj")?;
        ensure_biasless(shared_down_proj, "shared_down_proj")?;
        let weights = MetalMoeSharedWeights {
            router: self.resolve_linear_weight_buffers(router.weight(), "resident_moe_router")?,
            stacked: self.stacked_moe_buffers_with_context(experts, trace_context)?,
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
        };
        if crate::runtime_flags::trace_moe_enabled()
            || crate::runtime_flags::trace_resident_enabled()
        {
            trace_moe_shared_weights(&weights);
        }
        Ok(weights)
    }

    pub(crate) fn resolve_moe_routed_weights(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
    ) -> Result<MetalMoeRoutedWeights> {
        self.resolve_moe_routed_weights_with_context(
            router,
            experts,
            MoeStackTraceContext::runtime(),
        )
    }

    pub(crate) fn resolve_moe_routed_weights_for_cpu_release(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        layer_index: Option<usize>,
    ) -> Result<MetalMoeRoutedWeights> {
        self.resolve_moe_routed_weights_with_context(
            router,
            experts,
            MoeStackTraceContext::cpu_release(layer_index),
        )
    }

    fn resolve_moe_routed_weights_with_context(
        &self,
        router: &Linear,
        experts: &[GatedMlp],
        trace_context: MoeStackTraceContext,
    ) -> Result<MetalMoeRoutedWeights> {
        ensure_biasless(router, "router")?;
        Ok(MetalMoeRoutedWeights {
            router: self.resolve_linear_weight_buffers(router.weight(), "resident_moe_router")?,
            stacked: self.stacked_moe_buffers_with_context(experts, trace_context)?,
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

    /// Alloue le stockage partagé définitif d'un payload de poids quantifié.
    pub(crate) fn weight_buffer_from_u32(&self, data: &[u32]) -> Result<metal::Buffer> {
        self.buffer_from_u32(data, "single_copy_weight_packed")
    }

    /// Alloue le stockage partagé définitif d'un payload bf16 natif.
    pub(crate) fn weight_buffer_from_bf16_bytes(&self, data: &[u8]) -> Result<metal::Buffer> {
        if data.len() % std::mem::size_of::<u16>() != 0 {
            return Err(InferError::Shape(format!(
                "payload bf16 single-copy de {} octets",
                data.len()
            )));
        }
        self.buffer_from_slice(data, "single_copy_weight_affine")
    }

    /// Lit un payload bf16 directement dans son stockage partagé définitif.
    #[expect(
        unsafe_code,
        reason = "écriture initiale d'un MTLBuffer StorageModeShared depuis le fichier de poids"
    )]
    pub(crate) fn weight_buffer_from_bf16_file(
        &self,
        path: &std::path::Path,
        offset: u64,
        len_bytes: usize,
    ) -> Result<metal::Buffer> {
        use std::io::{Read, Seek};

        if len_bytes == 0 || len_bytes % std::mem::size_of::<u16>() != 0 {
            return Err(InferError::Shape(format!(
                "payload bf16 single-copy de {len_bytes} octets"
            )));
        }
        let buffer = self.device.new_buffer(
            checked_nsuint(len_bytes, "single_copy_weight_affine_file")?,
            MTLResourceOptions::StorageModeShared,
        );
        let destination = buffer.contents().cast::<u8>();
        if destination.is_null() {
            return Err(InferError::Metal(
                "MTLBuffer affine StorageModeShared sans pointeur CPU".to_string(),
            ));
        }
        // SAFETY: `buffer` vient d'être alloué en StorageModeShared avec exactement
        // `len_bytes`; son pointeur CPU est non nul et reste valide pendant la
        // lecture. Aucun command buffer GPU ne référence ce poids avant le retour.
        let destination = unsafe { std::slice::from_raw_parts_mut(destination, len_bytes) };
        let mut file = std::fs::File::open(path).map_err(|source| InferError::Io {
            path: path.to_path_buf(),
            source,
        })?;
        file.seek(std::io::SeekFrom::Start(offset))
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        file.read_exact(destination)
            .map_err(|source| InferError::Io {
                path: path.to_path_buf(),
                source,
            })?;
        Ok(buffer)
    }

    /// Alloue le stockage bf16 partagé définitif depuis une source f32 générique.
    pub(crate) fn weight_buffer_from_f32_as_bf16(&self, data: &[f32]) -> Result<metal::Buffer> {
        self.buffer_from_f32_as_bf16(data, "single_copy_weight_affine")
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
            MetalBufferSource::Pointer(data.as_ptr().addr()),
            data.len(),
            MetalBufferElement::F32,
            label,
            || self.buffer_from_f32(data, label),
        )
    }

    #[cfg(test)]
    pub(super) fn cached_buffer_from_u32(
        &self,
        data: &[u32],
        label: &'static str,
    ) -> Result<metal::Buffer> {
        self.cached_buffer(
            MetalBufferSource::Pointer(data.as_ptr().addr()),
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
            MetalBufferSource::Pointer(data.as_ptr().addr()),
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

    pub(super) fn cached_affine_packed(
        &self,
        weight: &AffineQuantizedTensor,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if let Some((buffer, _)) = weight.packed_metal_view() {
            return Ok(buffer.clone());
        }
        self.cached_buffer(
            MetalBufferSource::Affine {
                weight_id: weight.weight_id(),
                part: AffineWeightPart::Packed,
            },
            weight.packed_len(),
            MetalBufferElement::U32,
            label,
            || self.buffer_from_u32(weight.packed_data(), label),
        )
    }

    pub(super) fn cached_affine_scales(
        &self,
        weight: &AffineQuantizedTensor,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if let Some((buffer, _)) = weight.scales_metal_view() {
            return Ok(buffer.clone());
        }
        self.cached_buffer(
            MetalBufferSource::Affine {
                weight_id: weight.weight_id(),
                part: AffineWeightPart::Scales,
            },
            weight.scales_len(),
            MetalBufferElement::Bf16,
            label,
            || {
                let scales = weight.scales_f32();
                self.buffer_from_f32_as_bf16(scales.as_ref(), label)
            },
        )
    }

    pub(super) fn cached_affine_biases(
        &self,
        weight: &AffineQuantizedTensor,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if let Some((buffer, _)) = weight.biases_metal_view() {
            return Ok(buffer.clone());
        }
        self.cached_buffer(
            MetalBufferSource::Affine {
                weight_id: weight.weight_id(),
                part: AffineWeightPart::Biases,
            },
            weight.biases_len(),
            MetalBufferElement::Bf16,
            label,
            || {
                let biases = weight.biases_f32();
                self.buffer_from_f32_as_bf16(biases.as_ref(), label)
            },
        )
    }

    pub(super) fn cached_buffer(
        &self,
        source: MetalBufferSource,
        len: usize,
        element: MetalBufferElement,
        label: &'static str,
        create: impl FnOnce() -> Result<metal::Buffer>,
    ) -> Result<metal::Buffer> {
        let key = MetalBufferKey {
            source,
            len,
            element,
        };
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

    pub(super) fn new_bf16_buffer(&self, len: usize, label: &'static str) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer(len, MetalBufferElement::Bf16, label)
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

    /// Buffer u32 frais NON mémoïsé (StorageModeShared, lisible CPU) — pour les
    /// readbacks diagnostiques où la mémoïsation par label aliaserait les
    /// occurrences (indices d'experts par couche, stats light-batch).
    pub(super) fn uncached_u32_buffer(
        &self,
        len: usize,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        Ok(self
            .device
            .new_buffer(byte_len::<u32>(len)?, MTLResourceOptions::StorageModeShared))
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

    pub(super) fn private_bf16_buffer(
        &self,
        len: usize,
        label: &'static str,
    ) -> Result<metal::Buffer> {
        if len == 0 {
            return Err(InferError::Metal(format!("buffer {label} vide")));
        }
        self.scratch_buffer_with_options(
            len,
            MetalBufferElement::Bf16,
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
            namespace: current_scratch_namespace(),
        };
        let mut buffers = self
            .scratch_buffers
            .lock()
            .map_err(|_| InferError::Metal(format!("cache scratch Metal empoisonné: {label}")))?;
        let last_used = buffers.next_use();
        if let Some(entry) = buffers.entries.get_mut(&key) {
            entry.last_used = last_used;
            // Un cache miss fournit déjà un buffer au contenu indéfini : tous
            // les appelants scratch l'écrasent avant lecture. Un état précédent
            // `Empty` après pression mémoire respecte donc le même contrat.
            let _ = entry
                .buffer
                .set_purgeable_state(MTLPurgeableState::NonVolatile);
            return Ok(entry.buffer.clone());
        }
        let bytes = match element {
            MetalBufferElement::F32 => byte_len::<f32>(len)?,
            MetalBufferElement::U32 => byte_len::<u32>(len)?,
            MetalBufferElement::Bf16 => byte_len::<u16>(len)?,
        };
        let buffer = self.device.new_buffer(bytes, options);
        buffers.total_bytes = buffers.total_bytes.saturating_add(buffer.length());
        buffers.entries.insert(
            key,
            ScratchBufferEntry {
                buffer: buffer.clone(),
                last_used,
            },
        );
        Ok(buffer)
    }

    pub(crate) fn scratch_buffer_stats(&self) -> Result<(usize, u64)> {
        let buffers = self
            .scratch_buffers
            .lock()
            .map_err(|_| InferError::Metal("cache scratch Metal empoisonné".to_string()))?;
        Ok((buffers.entries.len(), buffers.total_bytes))
    }

    pub(crate) fn scratch_buffer_scope(&self) -> impl Drop + '_ {
        ScratchBufferScope {
            executor: self,
            namespace: current_scratch_namespace(),
        }
    }

    fn release_scratch_namespace(&self, namespace: u64) -> Result<()> {
        let cap_bytes = crate::runtime_flags::scratch_cap_bytes();
        let mut buffers = self
            .scratch_buffers
            .lock()
            .map_err(|_| InferError::Metal("cache scratch Metal empoisonné".to_string()))?;
        let before_entries = buffers.entries.len();
        let before_bytes = buffers.total_bytes;
        let mut volatile_entries = 0_usize;
        for (key, entry) in &mut buffers.entries {
            if key.namespace == namespace {
                // Le scope tombe après le prefill/decode complet. Marquer plus
                // tôt exposerait au purge un buffer encore référencé par une
                // command buffer Metal en vol.
                let _ = entry
                    .buffer
                    .set_purgeable_state(MTLPurgeableState::Volatile);
                volatile_entries = volatile_entries.saturating_add(1);
            }
        }

        let mut evicted_entries = 0_usize;
        let mut evicted_bytes = 0_u64;
        while buffers.total_bytes > cap_bytes {
            let Some(lru_key) = buffers
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key)
            else {
                break;
            };
            let Some(entry) = buffers.entries.remove(&lru_key) else {
                break;
            };
            let bytes = entry.buffer.length();
            buffers.total_bytes = buffers.total_bytes.saturating_sub(bytes);
            evicted_entries = evicted_entries.saturating_add(1);
            evicted_bytes = evicted_bytes.saturating_add(bytes);
        }

        if crate::runtime_flags::trace_gpu_alloc_enabled() {
            eprintln!(
                "scratch_cache_trim namespace={namespace} cap_bytes={cap_bytes} \
                 before_entries={before_entries} before_bytes={before_bytes} \
                 volatile_entries={volatile_entries} evicted_entries={evicted_entries} \
                 evicted_bytes={evicted_bytes} after_entries={} after_bytes={}",
                buffers.entries.len(),
                buffers.total_bytes,
            );
        }
        Ok(())
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

struct ScratchBufferScope<'a> {
    executor: &'a MetalExecutor,
    namespace: u64,
}

impl Drop for ScratchBufferScope<'_> {
    fn drop(&mut self) {
        if let Err(error) = self.executor.release_scratch_namespace(self.namespace) {
            eprintln!(
                "scratch_cache_trim namespace={} error={error}",
                self.namespace
            );
        }
    }
}

fn trace_linear_attn_pair(label: &str, pair: &MetalLinearAttnResidentPairWeights) {
    let message = match pair {
        MetalLinearAttnResidentPairWeights::Concat(weight) => {
            format!(
                "linear-attn resident pair {label}: concat {}",
                describe_linear_weight_buffers(weight)
            )
        }
        MetalLinearAttnResidentPairWeights::Split { first, second } => {
            format!(
                "linear-attn resident pair {label}: split first={} second={}",
                describe_linear_weight_buffers(first),
                describe_linear_weight_buffers(second)
            )
        }
    };
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut seen) = seen.lock() {
        if !seen.insert(message.clone()) {
            return;
        }
    }
    eprintln!("{message}");
}

fn trace_moe_shared_weights(weights: &MetalMoeSharedWeights) {
    let message = format!(
        "moe shared resident weights: router={} routed_gate={} routed_up={} routed_down={} \
         shared_gate={} shared_gate_proj={} shared_up_proj={} shared_down_proj={}",
        describe_linear_weight_buffers(&weights.router),
        describe_stacked_affine_buffers(&weights.stacked.gate),
        describe_stacked_affine_buffers(&weights.stacked.up),
        describe_stacked_affine_buffers(&weights.stacked.down),
        describe_linear_weight_buffers(&weights.shared_gate),
        describe_linear_weight_buffers(&weights.shared_gate_proj),
        describe_linear_weight_buffers(&weights.shared_up_proj),
        describe_linear_weight_buffers(&weights.shared_down_proj),
    );
    static SEEN: OnceLock<Mutex<std::collections::HashSet<String>>> = OnceLock::new();
    let seen = SEEN.get_or_init(|| Mutex::new(std::collections::HashSet::new()));
    if let Ok(mut seen) = seen.lock() {
        if !seen.insert(message.clone()) {
            return;
        }
    }
    eprintln!("{message}");
}

fn describe_stacked_affine_buffers(weight: &StackedAffineBuffers) -> String {
    format!(
        "affine bits={} gs={} groups={} experts={} out={} in={}",
        weight.bits,
        weight.group_size,
        weight.groups,
        weight.experts,
        weight.out_dim,
        weight.in_dim
    )
}

fn describe_linear_weight_buffers(weight: &MetalLinearWeightBuffers) -> String {
    match weight {
        MetalLinearWeightBuffers::Dense {
            out_dim, in_dim, ..
        } => format!("dense out={out_dim} in={in_dim}"),
        MetalLinearWeightBuffers::AffineQuantized {
            out_dim,
            in_dim,
            group_size,
            bits,
            groups,
            ..
        } => {
            format!("affine bits={bits} gs={group_size} groups={groups} out={out_dim} in={in_dim}")
        }
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
