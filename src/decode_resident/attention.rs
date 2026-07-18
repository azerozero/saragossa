//! Attention full-attn GPU-résidente et helpers RoPE/gating.

use super::arena::alloc_tensor;
use super::utils::{
    bf16_round_f32, byte_offset, flash_sdpa_enabled, kv_bf16_for, kv_bf16_sim_konly,
    kv_bf16_sim_vonly, sdpa_2pass_blocks, sdpa_2pass_enabled, sdpa_2pass_min_len,
    write_f32_as_bf16_at, write_f32_at,
};
use super::*;
use crate::metal_backend::EncoderEndGuard;

impl DecodeResidentState {
    /// Crée le KV-cache full-attn résident d'UNE couche (factory). L'état renvoyé
    /// est **auto-suffisant** : il embarque des clones bon marché de la queue, du
    /// pool scratch (partagé entre couches) et du pipeline d'attention → le decode
    /// résident des 10 couches n'a plus besoin de l'arène ensuite.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est nulle, si `q_heads` n'est pas un
    /// multiple de `kv_heads` (GQA, réserve R4), ou si la taille déborde.
    ///
    /// `sampled` = la génération courante échantillonne (temperature > 0) : pilote
    /// le dtype KV par défaut (bf16 échantillonné, f32 greedy) sauf override
    /// explicite `RETI_RUST_KV_BF16` (cf. [`kv_bf16_for`]).
    pub(crate) fn full_attention(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        sampled: bool,
    ) -> Result<FullAttentionMetalState> {
        let kv_element = if kv_bf16_for(sampled) {
            GpuElement::Bf16
        } else {
            GpuElement::F32
        };
        if crate::runtime_flags::trace_resident_enabled() {
            eprintln!("kv résident full-attn dtype={kv_element:?} (sampled={sampled})");
        }
        self.full_attention_with_element(capacity, q_heads, kv_heads, head_dim, kv_element)
    }

    #[cfg(test)]
    pub(crate) fn full_attention_bf16_for_test(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<FullAttentionMetalState> {
        self.full_attention_with_element(capacity, q_heads, kv_heads, head_dim, GpuElement::Bf16)
    }

    fn full_attention_with_element(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_element: GpuElement,
    ) -> Result<FullAttentionMetalState> {
        if capacity == 0 || q_heads == 0 || kv_heads == 0 || head_dim == 0 {
            return Err(InferError::Config(
                "FullAttentionMetalState: dimension nulle".to_string(),
            ));
        }
        // R4 : la GQA exige q_heads multiple de kv_heads (kv_group = q_heads/kv_heads).
        if q_heads % kv_heads != 0 {
            return Err(InferError::Config(format!(
                "GQA invalide: q_heads={q_heads} non multiple de kv_heads={kv_heads}"
            )));
        }
        let kv_dim = kv_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Config("kv_dim déborde".to_string()))?;
        let cells = capacity
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Config("capacité KV déborde".to_string()))?;
        let keys = alloc_tensor(&self.device, self.options, cells, kv_element)?;
        let values = alloc_tensor(&self.device, self.options, cells, kv_element)?;
        Ok(FullAttentionMetalState {
            keys,
            values,
            capacity,
            len: 0,
            q_heads,
            kv_heads,
            head_dim,
            queue: self.queue.clone(),
            scratch: self.scratch.clone(),
            attention_naive: self.attention_decode_naive.clone(),
            attention_flash: self.attention_decode_flash.clone(),
            attention_flash_d256: self.attention_decode_flash_d256.clone(),
            attention_2pass_1: self.attention_decode_2pass_1.clone(),
            attention_2pass_1_d128: self.attention_decode_2pass_1_d128.clone(),
            attention_naive_bf16: self.attention_decode_naive_bf16.clone(),
            attention_flash_bf16: self.attention_decode_flash_bf16.clone(),
            attention_flash_d256_bf16: self.attention_decode_flash_d256_bf16.clone(),
            attention_2pass_1_bf16: self.attention_decode_2pass_1_bf16.clone(),
            attention_2pass_1_d128_bf16: self.attention_decode_2pass_1_d128_bf16.clone(),
            attention_2pass_2: self.attention_decode_2pass_2.clone(),
            attention_2pass_2_d128: self.attention_decode_2pass_2_d128.clone(),
            copy_at: self.copy_at_kernel.clone(),
            copy_at_f32_to_bf16: self.copy_at_f32_to_bf16_kernel.clone(),
        })
    }

    /// Désinterleave la projection q_proj `[2·q_dim]` (gate de sortie full-attn) en
    /// `q [q_dim]` + `gate [q_dim]` sur GPU, en UN command buffer ; renvoie `(q, gate)`.
    /// Reproduit `split_attention_gate` (decoder.rs) sans readback au milieu. La
    /// variante encode (encoder partagé) sera extraite en 1c.2.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `proj.len() != 2·num_heads·head_dim` ou si une
    /// dimension déborde / l'exécution Metal échoue.
    pub(crate) fn split_q_gate(
        &self,
        proj: &[f32],
        num_heads: usize,
        head_dim: usize,
    ) -> Result<(Vec<f32>, Vec<f32>)> {
        let q_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("split q/gate q_dim déborde".to_string()))?;
        if proj.len() != 2 * q_dim {
            return Err(InferError::Dimension(format!(
                "split q/gate: proj attendu {}, reçu {}",
                2 * q_dim,
                proj.len()
            )));
        }
        let proj_lease = self.scratch.lease(2 * q_dim, GpuElement::F32)?;
        write_f32_at(proj_lease.tensor(), 0, proj)?;
        let q_lease = self.scratch.lease(q_dim, GpuElement::F32)?;
        let gate_lease = self.scratch.lease(q_dim, GpuElement::F32)?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_split_q_gate(
            encoder,
            proj_lease.tensor().buffer(),
            q_lease.tensor().buffer(),
            gate_lease.tensor().buffer(),
            num_heads,
            head_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        let q = read_f32_buffer(q_lease.tensor().buffer(), q_dim)?;
        let gate = read_f32_buffer(gate_lease.tensor().buffer(), q_dim)?;
        Ok((q, gate))
    }

    /// Encode le désinterleaving q/gate dans un encoder PARTAGÉ : `proj [2·q_dim]`
    /// résident → `q_out [q_dim]` + `gate_out [q_dim]` résidents, sans commit/readback.
    /// Cœur extrait de [`Self::split_q_gate`] (désormais wrapper), réutilisé par
    /// l'orchestration full-attn 1c.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde.
    pub(crate) fn encode_split_q_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proj: &BufferRef,
        q_out: &BufferRef,
        gate_out: &BufferRef,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        self.encode_split_q_gate_with_offset(encoder, proj, 0, q_out, gate_out, num_heads, head_dim)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "signature d'encodage Metal: buffers + offset + dimensions"
    )]
    pub(super) fn encode_split_q_gate_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        proj: &BufferRef,
        proj_offset: u64,
        q_out: &BufferRef,
        gate_out: &BufferRef,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        let q_dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("split q/gate q_dim déborde".to_string()))?;
        let dims: [u32; 2] = [
            u32::try_from(num_heads)
                .map_err(|_| InferError::Dimension("num_heads hors u32".to_string()))?,
            u32::try_from(head_dim)
                .map_err(|_| InferError::Dimension("head_dim hors u32".to_string()))?,
        ];
        encoder.set_compute_pipeline_state(&self.split_q_gate_kernel);
        encoder.set_buffer(0, Some(proj), proj_offset);
        encoder.set_buffer(1, Some(q_out), 0);
        encoder.set_buffer(2, Some(gate_out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 2]>() as u64,
            dims.as_ptr().cast::<c_void>(),
        );
        let width = self.split_q_gate_kernel.thread_execution_width().max(1);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_threads(MTLSize::new(q_dim as u64, 1, 1), MTLSize::new(width, 1, 1));
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Applique le gate de sortie full-attn `out = ctx · σ(gate)` sur GPU, en UN
    /// command buffer. Variante encode extraite en 1c.2.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `ctx`/`gate` ont des longueurs différentes ou nulles,
    /// ou si l'exécution Metal échoue.
    pub(crate) fn apply_attn_gate(&self, ctx: &[f32], gate: &[f32]) -> Result<Vec<f32>> {
        let n = ctx.len();
        if n == 0 || gate.len() != n {
            return Err(InferError::Dimension(format!(
                "attn_gate: ctx={n}, gate={}",
                gate.len()
            )));
        }
        let ctx_lease = self.scratch.lease(n, GpuElement::F32)?;
        write_f32_at(ctx_lease.tensor(), 0, ctx)?;
        let gate_lease = self.scratch.lease(n, GpuElement::F32)?;
        write_f32_at(gate_lease.tensor(), 0, gate)?;
        let out_lease = self.scratch.lease(n, GpuElement::F32)?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_attn_gate(
            encoder,
            ctx_lease.tensor().buffer(),
            gate_lease.tensor().buffer(),
            out_lease.tensor().buffer(),
            n,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(out_lease.tensor().buffer(), n)
    }

    /// Encode le gate de sortie `out = ctx · σ(gate)` dans un encoder PARTAGÉ
    /// (buffers résidents `[n]`), sans commit/readback. Cœur extrait de
    /// [`Self::apply_attn_gate`] (désormais wrapper), réutilisé par l'orchestration
    /// full-attn 1c.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `n` déborde `u32`.
    pub(crate) fn encode_attn_gate(
        &self,
        encoder: &ComputeCommandEncoderRef,
        ctx: &BufferRef,
        gate: &BufferRef,
        out: &BufferRef,
        n: usize,
    ) -> Result<()> {
        let len = u32::try_from(n)
            .map_err(|_| InferError::Dimension("attn_gate n hors u32".to_string()))?;
        encoder.set_compute_pipeline_state(&self.attn_gate_kernel);
        encoder.set_buffer(0, Some(ctx), 0);
        encoder.set_buffer(1, Some(gate), 0);
        encoder.set_buffer(2, Some(out), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<u32>() as u64,
            std::ptr::from_ref(&len).cast::<c_void>(),
        );
        let width = self.attn_gate_kernel.thread_execution_width().max(1);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(width, 1, 1));
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Encode rms_norm par tête + RoPE à la `position` du token courant (single-query)
    /// dans un encoder PARTAGÉ : `input [heads·head_dim]` → `output [heads·head_dim]`.
    /// `weight` = norme par tête `[head_dim]`. Reproduit `rms_norm_rope_heads_at`
    /// (decoder.rs) — rote à `position`, PAS à l'index de ligne.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde `u32`.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres rms_norm+RoPE (dims + position + theta)"
    )]
    pub(crate) fn encode_rms_norm_rope_decode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &BufferRef,
        output: &BufferRef,
        num_heads: usize,
        head_dim: usize,
        rope_dims: usize,
        position: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        self.encode_rms_norm_rope_decode_with_offset(
            encoder, input, 0, weight, output, num_heads, head_dim, rope_dims, position, eps,
            base_theta,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres rms_norm+RoPE (dims + position + theta)"
    )]
    pub(super) fn encode_rms_norm_rope_decode_with_offset(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        input_offset: u64,
        weight: &BufferRef,
        output: &BufferRef,
        num_heads: usize,
        head_dim: usize,
        rope_dims: usize,
        position: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        let dims: [u32; 4] = [
            u32::try_from(num_heads)
                .map_err(|_| InferError::Dimension("rope num_heads hors u32".to_string()))?,
            u32::try_from(head_dim)
                .map_err(|_| InferError::Dimension("rope head_dim hors u32".to_string()))?,
            u32::try_from(rope_dims)
                .map_err(|_| InferError::Dimension("rope rope_dims hors u32".to_string()))?,
            u32::try_from(position)
                .map_err(|_| InferError::Dimension("rope position hors u32".to_string()))?,
        ];
        let params: [f32; 2] = [eps, base_theta];
        encoder.set_compute_pipeline_state(&self.rope_decode_kernel);
        encoder.set_buffer(0, Some(input), input_offset);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 4]>() as u64,
            dims.as_ptr().cast::<c_void>(),
        );
        encoder.set_bytes(
            4,
            std::mem::size_of::<[f32; 2]>() as u64,
            params.as_ptr().cast::<c_void>(),
        );
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[0]), 1, 1),
            MTLSize::new(256, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Variante standalone (un command buffer) de [`Self::encode_rms_norm_rope_decode`]
    /// pour le test ==CPU. Renvoie le tenseur normé+roté `[num_heads·head_dim]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `input`/`weight` ont des longueurs incohérentes ou si
    /// l'exécution Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres rms_norm+RoPE"
    )]
    pub(crate) fn rms_norm_rope_decode(
        &self,
        input: &[f32],
        weight: &[f32],
        num_heads: usize,
        head_dim: usize,
        rope_dims: usize,
        position: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<Vec<f32>> {
        let dim = num_heads
            .checked_mul(head_dim)
            .ok_or_else(|| InferError::Dimension("rope dim déborde".to_string()))?;
        if input.len() != dim || weight.len() != head_dim {
            return Err(InferError::Dimension(format!(
                "rope decode: input={} (attendu {dim}), weight={} (attendu {head_dim})",
                input.len(),
                weight.len()
            )));
        }
        let in_lease = self.scratch.lease(dim, GpuElement::F32)?;
        write_f32_at(in_lease.tensor(), 0, input)?;
        let weight_lease = self.scratch.lease(head_dim, GpuElement::F32)?;
        write_f32_at(weight_lease.tensor(), 0, weight)?;
        let out_lease = self.scratch.lease(dim, GpuElement::F32)?;
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_rms_norm_rope_decode(
            encoder,
            in_lease.tensor().buffer(),
            weight_lease.tensor().buffer(),
            out_lease.tensor().buffer(),
            num_heads,
            head_dim,
            rope_dims,
            position,
            eps,
            base_theta,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        read_f32_buffer(out_lease.tensor().buffer(), dim)
    }
}

/// KV-cache full-attn **résident GPU** d'UNE couche (10 couches sur Qwen3.6).
///
/// Remplace, derrière le flag du decode résident, le `Vec<f32>` CPU append-only
/// (`decoder.rs` `LayerKvCache.keys/values`) et l'attention CPU
/// (`cached_attention_one`). Les buffers `keys`/`values` sont **persistants**
/// (alloués une fois via [`DecodeResidentState::persistent`], capacité bornée par
/// `prefill_len + max_new_tokens`) et restent GPU-résidents entre tokens.
///
/// **Non clonable** (réserve Codex D) : un état résident GPU est lié à une
/// session ; le `Clone` du cache englobant doit le remettre à `None` (drop des
/// buffers), jamais le partager — câblé en 1b.3.
#[derive(Debug)]
pub(crate) struct FullAttentionMetalState {
    keys: GpuTensor,
    values: GpuTensor,
    capacity: usize,
    len: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    // Ressources partagées (clones bon marché de l'arène) → état auto-suffisant.
    queue: CommandQueue,
    scratch: ScratchPool,
    attention_naive: ComputePipelineState,
    attention_flash: ComputePipelineState,
    attention_flash_d256: ComputePipelineState,
    attention_2pass_1: ComputePipelineState,
    attention_2pass_1_d128: ComputePipelineState,
    attention_naive_bf16: ComputePipelineState,
    attention_flash_bf16: ComputePipelineState,
    attention_flash_d256_bf16: ComputePipelineState,
    attention_2pass_1_bf16: ComputePipelineState,
    attention_2pass_1_d128_bf16: ComputePipelineState,
    attention_2pass_2: ComputePipelineState,
    attention_2pass_2_d128: ComputePipelineState,
    copy_at: ComputePipelineState,
    copy_at_f32_to_bf16: ComputePipelineState,
}

impl FullAttentionMetalState {
    /// Renvoie la dimension d'une ligne KV (`kv_heads * head_dim`).
    pub(crate) fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Renvoie le nombre de lignes valides (= position courante).
    pub(crate) fn len(&self) -> usize {
        self.len
    }

    /// Tronque le nombre de lignes K/V valides sans recopier les buffers.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len` dépasse la capacité allouée.
    pub(crate) fn truncate(&mut self, len: usize) -> Result<()> {
        if len > self.capacity {
            return Err(InferError::Dimension(format!(
                "truncate KV {len} > capacité {}",
                self.capacity
            )));
        }
        self.len = len;
        Ok(())
    }

    /// Renvoie la capacité en lignes.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Renvoie le buffer résident des clés (`capacity * kv_dim`).
    pub(crate) fn keys(&self) -> &GpuTensor {
        &self.keys
    }

    /// Renvoie le buffer résident des valeurs (`capacity * kv_dim`).
    pub(crate) fn values(&self) -> &GpuTensor {
        &self.values
    }

    /// Renvoie le nombre de têtes de requête.
    pub(crate) fn q_heads(&self) -> usize {
        self.q_heads
    }

    /// Renvoie le nombre de têtes clé/valeur.
    pub(crate) fn kv_heads(&self) -> usize {
        self.kv_heads
    }

    /// Renvoie la dimension d'une tête.
    pub(crate) fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Réinjecte `n_rows` lignes K/V (rope'd) issues du prefill, à l'offset 0, et
    /// pose `len = n_rows`. `keys_rows`/`values_rows` sont les `Vec` CPU du prefill
    /// (`[n_rows, kv_dim]`, ligne-major).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `n_rows > capacity` ou si les longueurs ne valent
    /// pas `n_rows * kv_dim`.
    pub(crate) fn seed(
        &mut self,
        keys_rows: &[f32],
        values_rows: &[f32],
        n_rows: usize,
    ) -> Result<()> {
        if n_rows > self.capacity {
            return Err(InferError::Dimension(format!(
                "seed KV {n_rows} > capacité {}",
                self.capacity
            )));
        }
        let kv_dim = self.kv_dim();
        let expected = n_rows
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("seed KV déborde".to_string()))?;
        if keys_rows.len() != expected || values_rows.len() != expected {
            return Err(InferError::Dimension(format!(
                "seed KV: attendu {expected} (={n_rows}×{kv_dim}), reçu keys={} values={}",
                keys_rows.len(),
                values_rows.len()
            )));
        }
        match self.keys.element() {
            GpuElement::F32 if kv_bf16_sim_konly() => {
                // Diagnostic C1B : K bf16 (seed), V f32 exact — buffers/kernels f32.
                let rounded: Vec<f32> = keys_rows.iter().copied().map(bf16_round_f32).collect();
                write_f32_at(&self.keys, 0, &rounded)?;
                write_f32_at(&self.values, 0, values_rows)?;
            }
            GpuElement::F32 if kv_bf16_sim_vonly() => {
                // Diagnostic C1B : V bf16 (seed), K f32 exact.
                let rounded: Vec<f32> = values_rows.iter().copied().map(bf16_round_f32).collect();
                write_f32_at(&self.keys, 0, keys_rows)?;
                write_f32_at(&self.values, 0, &rounded)?;
            }
            GpuElement::F32 => {
                write_f32_at(&self.keys, 0, keys_rows)?;
                write_f32_at(&self.values, 0, values_rows)?;
            }
            GpuElement::Bf16 => {
                write_f32_as_bf16_at(&self.keys, 0, keys_rows)?;
                write_f32_as_bf16_at(&self.values, 0, values_rows)?;
            }
            GpuElement::U32 => {
                return Err(InferError::Metal(
                    "seed KV sur un buffer u32 invalide".to_string(),
                ));
            }
        }
        self.len = n_rows;
        Ok(())
    }

    /// Append la K/V (rope'd) du token courant à la ligne `len`, sans readback
    /// (écriture résidente). Incrémente `len`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `key`/`value` ne valent pas `kv_dim` ou si la
    /// capacité est atteinte (réserve R5 : pas de corruption silencieuse).
    pub(crate) fn append_row(&mut self, key: &[f32], value: &[f32]) -> Result<()> {
        let kv_dim = self.kv_dim();
        if key.len() != kv_dim || value.len() != kv_dim {
            return Err(InferError::Dimension(format!(
                "append KV: attendu {kv_dim}, reçu key={} value={}",
                key.len(),
                value.len()
            )));
        }
        if self.len >= self.capacity {
            return Err(InferError::Dimension(format!(
                "append KV: capacité {} atteinte (overflow)",
                self.capacity
            )));
        }
        let offset = self.len * kv_dim;
        match self.keys.element() {
            GpuElement::F32 => {
                write_f32_at(&self.keys, offset, key)?;
                write_f32_at(&self.values, offset, value)?;
            }
            GpuElement::Bf16 => {
                write_f32_as_bf16_at(&self.keys, offset, key)?;
                write_f32_as_bf16_at(&self.values, offset, value)?;
            }
            GpuElement::U32 => {
                return Err(InferError::Metal(
                    "append KV sur un buffer u32 invalide".to_string(),
                ));
            }
        }
        self.len += 1;
        Ok(())
    }

    /// Attention decode single-query sur le KV résident (lignes `0..len`), en UN
    /// command buffer. `q` est la requête CPU rope'd du token courant (`[q_dim]`).
    /// Renvoie le **contexte brut** `[q_dim]` (le gate de sortie full-attn reste
    /// appliqué par l'appelant, hors kernel).
    ///
    /// Réserve **R1** (liveness) rendue structurelle : `q_buf`, `scores` et `out`
    /// sont des bails ([`ScratchLease`]) **locaux à ce scope**, vivants jusqu'APRÈS
    /// le `wait` ; la fonction ne renvoie qu'un `Vec<f32>`, jamais un `GpuTensor`
    /// ni un bail → aucun buffer scratch ne survit à son bail (pas de réutilisation
    /// du pool pendant que le GPU le référence encore).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `q` n'a pas la longueur `q_dim`, si le KV est vide, ou
    /// si une dimension déborde / l'exécution Metal échoue.
    pub(crate) fn attention_decode(&self, q: &[f32]) -> Result<Vec<f32>> {
        self.attention_decode_windowed(q, 0)
    }

    /// Calcule l'attention decode sur les lignes KV `window_start..len`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la fenêtre est vide, une dimension déborde ou
    /// l'exécution Metal échoue.
    pub(crate) fn attention_decode_windowed(
        &self,
        q: &[f32],
        window_start: usize,
    ) -> Result<Vec<f32>> {
        if self.len == 0 {
            return Err(InferError::Dimension(
                "attention_decode sur KV vide".to_string(),
            ));
        }
        let q_dim = self
            .q_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| InferError::Dimension("attention_decode q_dim déborde".to_string()))?;
        if q.len() != q_dim {
            return Err(InferError::Dimension(format!(
                "attention_decode: q attendu [{q_dim}], reçu {}",
                q.len()
            )));
        }
        // R1 : bails LOCAUX (q uploadé, scores, out), vivants jusqu'après le wait ;
        // rien ne s'échappe. Scores dimensionnés à la CAPACITÉ (stride = len passé
        // au kernel) → buffer de taille fixe réutilisé par le pool à chaque token.
        let score_cells = self.q_heads.checked_mul(self.capacity).ok_or_else(|| {
            InferError::Dimension("attention_decode scores débordent".to_string())
        })?;
        let q_buf = self.scratch.lease(q_dim, GpuElement::F32)?;
        write_f32_at(q_buf.tensor(), 0, q)?;
        let scores = self.scratch.lease(score_cells, GpuElement::F32)?;
        let out = self.scratch.lease(q_dim, GpuElement::F32)?;

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_attention_decode_windowed(
            encoder,
            q_buf.tensor().buffer(),
            scores.tensor().buffer(),
            out.tensor().buffer(),
            window_start,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
        let context = read_f32_buffer(out.tensor().buffer(), q_dim)?;
        // `q_buf`, `scores`, `out` (bails) droppés ICI, après le readback.
        Ok(context)
    }

    /// Encode l'attention decode single-query dans un encoder PARTAGÉ : lit `q_buf`
    /// résident + le KV résident (`0..len`), écrit le contexte brut dans `out_buf`
    /// (`scores_buf` = scratch device, taille `q_heads·capacity`). Aucun commit ni
    /// readback → réutilisé par l'orchestration full-attn 1c. Cœur extrait de
    /// [`Self::attention_decode`] (désormais wrapper).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le KV est vide ou si une dimension déborde.
    pub(crate) fn encode_attention_decode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
    ) -> Result<()> {
        self.encode_attention_decode_windowed(encoder, q_buf, scores_buf, out_buf, 0)
    }

    pub(crate) fn encode_attention_decode_windowed(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
        window_start: usize,
    ) -> Result<()> {
        if self.len == 0 {
            return Err(InferError::Dimension(
                "attention_decode sur KV vide".to_string(),
            ));
        }
        if window_start >= self.len {
            return Err(InferError::Dimension(format!(
                "attention_decode fenêtre vide: début={window_start}, len={}",
                self.len
            )));
        }
        // Bascule 2-passes split-K pour les longs KV (dédup GQA + tuiles L1) :
        // head_dim 128/256, GQA réel (>1 q/kv), len ≥ seuil. Cf. encode_attention_decode_2pass.
        let gqa = self.q_heads / self.kv_heads.max(1);
        if window_start == 0
            && sdpa_2pass_enabled()
            && flash_sdpa_enabled()
            && matches!(self.head_dim, 128 | 256)
            && gqa > 1
            && self.q_heads % self.kv_heads == 0
            && self.len >= sdpa_2pass_min_len()
        {
            return self.encode_attention_decode_2pass(encoder, q_buf, out_buf);
        }
        let dims: [u32; 4] = [
            u32::try_from(self.q_heads)
                .map_err(|_| InferError::Dimension("q_heads hors u32".to_string()))?,
            u32::try_from(self.kv_heads)
                .map_err(|_| InferError::Dimension("kv_heads hors u32".to_string()))?,
            u32::try_from(self.head_dim)
                .map_err(|_| InferError::Dimension("head_dim hors u32".to_string()))?,
            u32::try_from(self.len)
                .map_err(|_| InferError::Dimension("len hors u32".to_string()))?,
        ];
        // Le chemin fenêtré privilégie d'abord la correction : le kernel naïf
        // exclut réellement les lignes antérieures. Les variantes flash/2-pass
        // historiques restent strictement inchangées pour `window_start == 0`.
        let use_flash = window_start == 0
            && flash_sdpa_enabled()
            && self.head_dim <= 256
            && self.head_dim % 32 == 0;
        let pipeline = match (self.keys.element(), use_flash, self.head_dim == 256) {
            (GpuElement::F32, true, true) => &self.attention_flash_d256,
            (GpuElement::F32, true, false) => &self.attention_flash,
            (GpuElement::F32, false, _) => &self.attention_naive,
            (GpuElement::Bf16, true, true) => &self.attention_flash_d256_bf16,
            (GpuElement::Bf16, true, false) => &self.attention_flash_bf16,
            (GpuElement::Bf16, false, _) => &self.attention_naive_bf16,
            (GpuElement::U32, _, _) => {
                return Err(InferError::Metal(
                    "attention_decode sur KV u32 invalide".to_string(),
                ));
            }
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(self.keys.buffer()), 0);
        encoder.set_buffer(2, Some(self.values.buffer()), 0);
        if use_flash {
            encoder.set_buffer(3, Some(out_buf), 0);
            encoder.set_bytes(
                4,
                std::mem::size_of::<[u32; 4]>() as u64,
                dims.as_ptr().cast::<c_void>(),
            );
        } else {
            encoder.set_buffer(3, Some(scores_buf), 0);
            encoder.set_buffer(4, Some(out_buf), 0);
            encoder.set_bytes(
                5,
                std::mem::size_of::<[u32; 4]>() as u64,
                dims.as_ptr().cast::<c_void>(),
            );
            let window_start = u32::try_from(window_start)
                .map_err(|_| InferError::Dimension("window_start hors u32".to_string()))?;
            encoder.set_bytes(
                6,
                std::mem::size_of::<u32>() as u64,
                std::ptr::from_ref(&window_start).cast::<c_void>(),
            );
        }
        crate::metal_backend::profile_dispatch();
        let threads = if use_flash { 1024 } else { 256 };
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[0]), 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// SDPA decode 2-passes split-K (head_dim 128/256) dans l'encoder partagé. Passe 1 :
    /// un threadgroup par `(kv_head, bloc)`, `gqa` simdgroups (un q_head chacun) → les
    /// lectures KV redondantes du groupe GQA touchent le L1, pas la DRAM ; la longueur
    /// est tuilée en `blocks`. Passe 2 : réduit les `blocks` partiels par q_head vers
    /// `out_buf`. Partials/sums/maxs = scratch loué (vivant jusqu'au wait, hazard-ordonné).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si l'allocation scratch échoue.
    fn encode_attention_decode_2pass(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        out_buf: &BufferRef,
    ) -> Result<()> {
        let blocks = sdpa_2pass_blocks(self.len);
        let gqa = self.q_heads / self.kv_heads;
        let reduce_cells = self
            .q_heads
            .checked_mul(blocks)
            .ok_or_else(|| InferError::Dimension("2pass reduce cells déborde".to_string()))?;
        let partial_cells = reduce_cells
            .checked_mul(self.head_dim)
            .ok_or_else(|| InferError::Dimension("2pass partials cells déborde".to_string()))?;
        let partials = self.scratch.lease(partial_cells, GpuElement::F32)?;
        let sums = self.scratch.lease(reduce_cells, GpuElement::F32)?;
        let maxs = self.scratch.lease(reduce_cells, GpuElement::F32)?;

        let dims: [u32; 4] = [
            u32::try_from(self.q_heads)
                .map_err(|_| InferError::Dimension("2pass q_heads hors u32".to_string()))?,
            u32::try_from(self.kv_heads)
                .map_err(|_| InferError::Dimension("2pass kv_heads hors u32".to_string()))?,
            u32::try_from(self.len)
                .map_err(|_| InferError::Dimension("2pass len hors u32".to_string()))?,
            u32::try_from(blocks)
                .map_err(|_| InferError::Dimension("2pass blocks hors u32".to_string()))?,
        ];
        // Passe 1 : partiels par (kv_head, bloc).
        let pass1 = match (self.keys.element(), self.head_dim) {
            (GpuElement::F32, 256) => &self.attention_2pass_1,
            (GpuElement::F32, 128) => &self.attention_2pass_1_d128,
            (GpuElement::Bf16, 256) => &self.attention_2pass_1_bf16,
            (GpuElement::Bf16, 128) => &self.attention_2pass_1_d128_bf16,
            (GpuElement::F32 | GpuElement::Bf16, other) => {
                return Err(InferError::Metal(format!(
                    "attention_decode 2pass head_dim={other} non supporté"
                )));
            }
            (GpuElement::U32, _) => {
                return Err(InferError::Metal(
                    "attention_decode 2pass sur KV u32 invalide".to_string(),
                ));
            }
        };
        encoder.set_compute_pipeline_state(pass1);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(self.keys.buffer()), 0);
        encoder.set_buffer(2, Some(self.values.buffer()), 0);
        encoder.set_buffer(3, Some(partials.tensor().buffer()), 0);
        encoder.set_buffer(4, Some(sums.tensor().buffer()), 0);
        encoder.set_buffer(5, Some(maxs.tensor().buffer()), 0);
        encoder.set_bytes(
            6,
            std::mem::size_of::<[u32; 4]>() as u64,
            dims.as_ptr().cast::<c_void>(),
        );
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[1]), u64::from(dims[3]), 1),
            MTLSize::new(32, u64::try_from(gqa).unwrap_or(1), 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);

        // Passe 2 : réduction des blocs par q_head.
        let blocks_u32 = dims[3];
        let pass2 = match self.head_dim {
            256 => &self.attention_2pass_2,
            128 => &self.attention_2pass_2_d128,
            other => {
                return Err(InferError::Metal(format!(
                    "attention_decode 2pass pass2 head_dim={other} non supporté"
                )));
            }
        };
        encoder.set_compute_pipeline_state(pass2);
        encoder.set_buffer(0, Some(partials.tensor().buffer()), 0);
        encoder.set_buffer(1, Some(sums.tensor().buffer()), 0);
        encoder.set_buffer(2, Some(maxs.tensor().buffer()), 0);
        encoder.set_buffer(3, Some(out_buf), 0);
        encoder.set_bytes(
            4,
            std::mem::size_of::<u32>() as u64,
            std::ptr::from_ref(&blocks_u32).cast::<c_void>(),
        );
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(dims[0]), 1, 1),
            MTLSize::new(32, 32, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Append device-side de la K/V (rope'd) du token courant : copie `k_buf`/`v_buf`
    /// résidents dans `keys`/`values` à l'offset `len·kv_dim` (écriture device, hazard
    /// read-after-write prouvé en R3), puis incrémente `len`. À encoder AVANT
    /// `encode_attention_decode` (qui lit `0..len`). Les longueurs de `k_buf`/`v_buf`
    /// (= `kv_dim`) sont garanties par l'appelant (buffers résidents).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur (overflow) si la capacité KV est atteinte.
    pub(crate) fn encode_append_kv(
        &mut self,
        encoder: &ComputeCommandEncoderRef,
        k_buf: &BufferRef,
        v_buf: &BufferRef,
    ) -> Result<()> {
        self.encode_append_kv_with_offsets(encoder, k_buf, 0, v_buf, 0)
    }

    pub(crate) fn encode_append_kv_with_offsets(
        &mut self,
        encoder: &ComputeCommandEncoderRef,
        k_buf: &BufferRef,
        k_offset: u64,
        v_buf: &BufferRef,
        v_offset: u64,
    ) -> Result<()> {
        if self.len >= self.capacity {
            return Err(InferError::Dimension(format!(
                "append KV résident: capacité {} atteinte (overflow)",
                self.capacity
            )));
        }
        let kv_dim = self.kv_dim();
        let offset = self.len * kv_dim;
        self.encode_copy_at(
            encoder,
            k_buf,
            k_offset,
            self.keys.buffer(),
            self.keys.element(),
            offset,
            kv_dim,
        )?;
        self.encode_copy_at(
            encoder,
            v_buf,
            v_offset,
            self.values.buffer(),
            self.values.element(),
            offset,
            kv_dim,
        )?;
        self.len += 1;
        Ok(())
    }

    /// Copie `input[0..n]` → `output[offset..offset+n]` (device, `output` lié à un
    /// offset en octets). Brique de l'append KV résident.
    fn encode_copy_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        input_offset: u64,
        output: &BufferRef,
        output_element: GpuElement,
        output_offset: usize,
        n: usize,
    ) -> Result<()> {
        let len = u32::try_from(n)
            .map_err(|_| InferError::Dimension("copy_at n hors u32".to_string()))?;
        let offset_bytes = byte_offset(output_offset, output_element, "copy_at offset")?;
        let pipeline = match output_element {
            GpuElement::F32 => &self.copy_at,
            GpuElement::Bf16 => &self.copy_at_f32_to_bf16,
            GpuElement::U32 => {
                return Err(InferError::Metal(
                    "copy_at KV vers un buffer u32 invalide".to_string(),
                ));
            }
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(input), input_offset);
        encoder.set_buffer(1, Some(output), offset_bytes);
        encoder.set_bytes(
            2,
            std::mem::size_of::<u32>() as u64,
            std::ptr::from_ref(&len).cast::<c_void>(),
        );
        let width = pipeline.thread_execution_width().max(1);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(width, 1, 1));
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }
}
