//! Attention full-attn GPU-résidente et helpers RoPE/gating.

use super::arena::alloc_tensor;
use super::gpu_position::{
    check_gpu_position_rows, rope_dims_param, to_u32, GpuPositionRows, U32x4Param,
};
use super::kv_quant::{KvEncoding, KvQuant};
use super::utils::{
    bf16_round_f32, byte_offset, flash_sdpa_enabled, kv_bf16_sim_konly, kv_bf16_sim_vonly,
    kv_format_for, kv_quant_refusal, read_f32_at, sdpa_2pass_blocks, sdpa_2pass_enabled,
    sdpa_2pass_min_len, write_f32_as_bf16_at, write_f32_at,
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
    /// `kv_bf16` = dtype KV par défaut choisi par l'appelant (bf16 si vrai),
    /// sauf override explicite `SARAGOSSA_RUST_KV_BF16` (cf. [`kv_format_for`]).
    pub(crate) fn full_attention(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_bf16: bool,
    ) -> Result<FullAttentionMetalState> {
        let requested = kv_format_for(kv_bf16);
        // Repli bf16 décidé AVANT allocation pour une couche sans attention
        // quantifiée (les fenêtres glissantes passent par `full_attention_dense`).
        let refusal = requested
            .is_quantized()
            .then(|| {
                kv_quant_refusal(
                    q_heads,
                    kv_heads,
                    head_dim,
                    flash_sdpa_enabled(),
                    sdpa_2pass_enabled(),
                )
            })
            .flatten();
        let format = if refusal.is_some() {
            KvFormat::Bf16
        } else {
            requested
        };
        if let Some(reason) = refusal {
            // Signalé une fois : un KV_QUANT demandé mais ignoré passerait
            // sinon inaperçu hors trace.
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!("SARAGOSSA_RUST_KV_QUANT ignoré ({reason}) : KV résident en bf16");
            });
        }
        if crate::runtime_flags::trace_resident_enabled() {
            eprintln!(
                "kv résident full-attn format={format:?} (kv_bf16={kv_bf16}, demandé {requested:?}{})",
                refusal.map_or(String::new(), |reason| format!(", repli: {reason}"))
            );
        }
        self.full_attention_with_format(capacity, q_heads, kv_heads, head_dim, format, None)
    }

    /// Comme [`Self::full_attention`], mais jamais quantifié : f32 ou bf16
    /// seulement. Sert aux arènes dont l'attention n'a pas de variante
    /// quantifiée (fenêtre glissante, tête MTP, DFlash) ou dont les octets sont
    /// comparés tels quels (oracles).
    ///
    /// # Errors
    ///
    /// Mêmes erreurs que [`Self::full_attention`].
    pub(crate) fn full_attention_dense(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_bf16: bool,
    ) -> Result<FullAttentionMetalState> {
        let format = match kv_format_for(kv_bf16) {
            KvFormat::F32 => KvFormat::F32,
            KvFormat::Bf16 | KvFormat::Q8 | KvFormat::K8V4 | KvFormat::Q4 => KvFormat::Bf16,
        };
        self.full_attention_with_format(capacity, q_heads, kv_heads, head_dim, format, None)
    }

    #[cfg(test)]
    pub(crate) fn full_attention_bf16_for_test(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<FullAttentionMetalState> {
        self.full_attention_with_format(capacity, q_heads, kv_heads, head_dim, KvFormat::Bf16, None)
    }

    #[cfg(test)]
    pub(crate) fn full_attention_format_for_test(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvFormat,
    ) -> Result<FullAttentionMetalState> {
        self.full_attention_with_format(capacity, q_heads, kv_heads, head_dim, format, None)
    }

    #[cfg(test)]
    pub(crate) fn full_attention_f32_for_test(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
    ) -> Result<FullAttentionMetalState> {
        self.full_attention_with_format(capacity, q_heads, kv_heads, head_dim, KvFormat::F32, None)
    }

    /// Comme [`Self::full_attention`], sur les buffers K/V bf16 d'un prefill
    /// au lieu d'une allocation neuve : les lignes de `prefix` (vues K et V
    /// du préfixe, depuis leur début) sont déjà en place et l'arène écrit à
    /// la suite. Renvoie `None` (l'appelant alloue et sème) si le format
    /// résolu n'est pas bf16 dense, si les vues ne sont pas de même longueur
    /// en lignes entières, si les buffers n'ont pas la place de `capacity`
    /// lignes, si un autre écrivain tient déjà leur réserve ou si, une fois
    /// la réserve prise, une vue vivante la couvre
    /// ([`crate::metal_backend::InPlaceClaim::try_after`]).
    ///
    /// # Errors
    ///
    /// Mêmes erreurs que [`Self::full_attention`].
    pub(crate) fn full_attention_adopting(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        kv_bf16: bool,
        prefix: [&crate::metal_backend::SharedBf16Rows; 2],
    ) -> Result<Option<FullAttentionMetalState>> {
        let kv_dim = kv_heads * head_dim;
        let [key_len, value_len] = prefix.map(crate::metal_backend::SharedBf16Rows::len);
        if key_len != value_len || kv_dim == 0 || key_len % kv_dim != 0 {
            return Ok(None);
        }
        let floor = key_len / kv_dim;
        if !kv_bf16_dense(kv_bf16) || floor > capacity {
            return Ok(None);
        }
        let buffers = prefix.map(|view| view.buffer().to_owned());
        let bytes = kv_heads
            .checked_mul(head_dim)
            .and_then(|kv_dim| kv_dim.checked_mul(capacity))
            .and_then(|cells| cells.checked_mul(GpuElement::Bf16.byte_size()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| InferError::Config("capacité KV adoptée déborde".to_string()))?;
        if buffers.iter().any(|buffer| buffer.length() < bytes) {
            return Ok(None);
        }
        // Claim pris puis revalidé : aucune vue vivante (snapshot plus long,
        // prolongement en place publié) ne couvre les lignes que l'arène
        // écrira.
        let Some(claim) = crate::metal_backend::InPlaceClaim::try_after(prefix) else {
            return Ok(None);
        };
        let mut state = self.full_attention_with_format(
            capacity,
            q_heads,
            kv_heads,
            head_dim,
            KvFormat::Bf16,
            Some(buffers),
        )?;
        state.len = floor;
        state.floor = floor;
        state.adopted = Some(AdoptedKv {
            queue: self.queue.clone(),
            _claim: claim,
        });
        Ok(Some(state))
    }

    fn full_attention_with_format(
        &self,
        capacity: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        format: KvFormat,
        existing: Option<[Buffer; 2]>,
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
        // Seule l'attention 2-passes d128/d256 lit le KV quantifié.
        let quant = if format.is_quantized() {
            Some(self.kv_quant(format, head_dim, cells)?)
        } else {
            None
        };
        let payload = |encoding: KvEncoding| {
            let len = encoding.payload_len(cells).ok_or_else(|| {
                InferError::Config(format!("KV {format:?}: {cells} cellules non groupées"))
            })?;
            alloc_tensor(&self.device, self.options, len, encoding.element())
        };
        let (keys, values) = match existing {
            Some([keys, values]) if format == KvFormat::Bf16 => (
                GpuTensor::adopt(keys, cells, GpuElement::Bf16)?,
                GpuTensor::adopt(values, cells, GpuElement::Bf16)?,
            ),
            Some(_) => {
                return Err(InferError::Config(format!(
                    "KV adopté en {format:?} : seul le bf16 dense l'est"
                )))
            }
            None => (payload(format.keys())?, payload(format.values())?),
        };
        Ok(FullAttentionMetalState {
            keys,
            values,
            format,
            quant,
            adopted: None,
            floor: 0,
            base: 0,
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
            copy_at_bf16: self.copy_at_bf16_kernel.clone(),
            copy_at_bf16_to_f32: self.copy_at_bf16_to_f32_kernel.clone(),
            sinks: None,
        })
    }

    /// Compile (une fois) les kernels d'attention avec sinks.
    fn sink_kernels(&self) -> Result<&SinkKernels> {
        if let Some(kernels) = self.sink_kernels.get() {
            return Ok(kernels);
        }
        let mut pipelines = super::kernels::compile_kernels(
            &self.device,
            super::kernels::ATTENTION_SINK_KERNELS,
            &[
                "attention_decode_naive_sink_f32",
                "attention_decode_naive_sink_bf16",
                "attention_decode_2pass_2_d128_sink_f32",
            ],
            true,
        )?
        .into_iter();
        let mut next = || {
            pipelines
                .next()
                .ok_or_else(|| InferError::Metal("pipeline sink absent".to_string()))
        };
        let kernels = SinkKernels {
            naive_f32: next()?,
            naive_bf16: next()?,
            two_pass_2_d128: next()?,
        };
        Ok(self.sink_kernels.get_or_init(|| kernels))
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
        rope_frequency_dim: usize,
        position: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        self.encode_rms_norm_rope_decode_with_offset(
            encoder,
            input,
            0,
            weight,
            output,
            num_heads,
            head_dim,
            rope_dims,
            rope_frequency_dim,
            position,
            eps,
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
        rope_frequency_dim: usize,
        position: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        let dims = rope_dims_param(num_heads, head_dim, rope_dims, position)?;
        self.encode_rms_norm_rope_decode_param(
            encoder,
            input,
            input_offset,
            weight,
            output,
            num_heads,
            dims,
            rope_frequency_dim,
            eps,
            base_theta,
        )
    }

    /// Cœur RoPE decode : `dims = [heads, head_dim, rope_dims, position]` lié par
    /// octets hôte (chemin historique) ou par buffer GPU (position GPU).
    #[expect(
        clippy::too_many_arguments,
        reason = "mirror des paramètres rms_norm+RoPE (dims liés + theta)"
    )]
    pub(super) fn encode_rms_norm_rope_decode_param(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        input_offset: u64,
        weight: &BufferRef,
        output: &BufferRef,
        num_heads: usize,
        dims: U32x4Param<'_>,
        rope_frequency_dim: usize,
        eps: f32,
        base_theta: f32,
    ) -> Result<()> {
        let params: [f32; 2] = [eps, base_theta];
        encoder.set_compute_pipeline_state(&self.rope_decode_kernel);
        encoder.set_buffer(0, Some(input), input_offset);
        encoder.set_buffer(1, Some(weight), 0);
        encoder.set_buffer(2, Some(output), 0);
        dims.bind(encoder, 3);
        encoder.set_bytes(
            4,
            std::mem::size_of::<[f32; 2]>() as u64,
            params.as_ptr().cast::<c_void>(),
        );
        let rope_frequency_dim = u32::try_from(rope_frequency_dim)
            .map_err(|_| InferError::Dimension("rope frequency_dim hors u32".to_string()))?;
        encoder.set_bytes(
            5,
            std::mem::size_of::<u32>() as u64,
            std::ptr::from_ref(&rope_frequency_dim).cast::<c_void>(),
        );
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(to_u32(num_heads, "rope num_heads")?), 1, 1),
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

/// Bloc de lignes K ou V (semis de l'arène, préfixe d'un prolongement) :
/// tranche CPU, ou vue d'un buffer GPU partagé copiée sur le GPU, en f32 ou en
/// bits bf16.
#[derive(Clone, Copy, Debug)]
pub(crate) enum KvSegment<'a> {
    Cpu(&'a [f32]),
    CpuBf16(&'a [u16]),
    Shared(&'a crate::metal_backend::SharedF32Rows),
    SharedBf16(&'a crate::metal_backend::SharedBf16Rows),
}

impl<'a> KvSegment<'a> {
    /// Valeurs en f32 : empruntées, ou élargies (exact) depuis le bf16.
    pub(crate) fn as_f32(&self) -> std::borrow::Cow<'a, [f32]> {
        let bf16 = match *self {
            Self::Cpu(data) => return std::borrow::Cow::Borrowed(data),
            Self::Shared(rows) => return std::borrow::Cow::Borrowed(rows.as_slice()),
            Self::CpuBf16(data) => data,
            Self::SharedBf16(rows) => rows.as_slice(),
        };
        std::borrow::Cow::Owned(
            bf16.iter()
                .map(|&bits| f32::from_bits(u32::from(bits) << 16))
                .collect(),
        )
    }

    /// Nombre d'éléments, sans lecture des données.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Cpu(data) => data.len(),
            Self::CpuBf16(data) => data.len(),
            Self::Shared(rows) => rows.len(),
            Self::SharedBf16(rows) => rows.len(),
        }
    }

    /// Buffer partagé source, octet de départ et élément, pour une copie GPU.
    pub(crate) fn shared(&self) -> Option<(&'a metal::BufferRef, u64, GpuElement)> {
        match *self {
            Self::Shared(rows) => Some((rows.buffer(), rows.byte_offset(), GpuElement::F32)),
            Self::SharedBf16(rows) => Some((rows.buffer(), rows.byte_offset(), GpuElement::Bf16)),
            Self::Cpu(_) | Self::CpuBf16(_) => None,
        }
    }
}

/// Réserve d'un buffer de prefill adopté par une arène. Le drop attend la
/// fin des commandes déjà soumises sur la file du decode (exécutées dans
/// l'ordre de soumission) avant de rendre le droit d'écrire : une sortie sur
/// erreur peut laisser des pas de decode en vol.
#[derive(Debug)]
struct AdoptedKv {
    queue: CommandQueue,
    _claim: crate::metal_backend::InPlaceClaim,
}

impl Drop for AdoptedKv {
    fn drop(&mut self) {
        let command_buffer = self.queue.new_command_buffer();
        command_buffer.commit();
        command_buffer.wait_until_completed();
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
    /// Charge utile : f32, bf16, ou codes u8 d'un format quantifié.
    keys: GpuTensor,
    values: GpuTensor,
    format: KvFormat,
    /// Échelles et kernels d'écriture, présents si et seulement si `format`
    /// est quantifié.
    quant: Option<KvQuant>,
    /// Buffers K/V d'un prefill adoptés : droit exclusif sur leur réserve,
    /// rendu après la fin des commandes de la file.
    adopted: Option<AdoptedKv>,
    /// Lignes du préfixe adopté, vues par le cache CPU : jamais réécrites,
    /// `len` ne descend pas en dessous.
    floor: usize,
    /// Lignes absolues absentes devant la ligne 0 du buffer (couche
    /// glissante bornée à sa fenêtre, cf. [`Self::rebase`]). `capacity`,
    /// `len` et `floor` comptent les lignes du buffer ; l'API publique
    /// (`len()`, `truncate`, `read_rows`, `seed_segments`, fenêtre) parle en
    /// lignes absolues.
    base: usize,
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
    copy_at_bf16: ComputePipelineState,
    copy_at_bf16_to_f32: ComputePipelineState,
    /// Sinks appris par tête (flint) et leurs kernels ; `None` = attention
    /// historique, kernels sans sink inchangés.
    sinks: Option<AttentionSinks>,
}

/// Pipelines de l'attention decode avec sinks (source distincte, cf.
/// `ATTENTION_SINK_KERNELS`).
#[derive(Clone, Debug)]
pub(crate) struct SinkKernels {
    naive_f32: ComputePipelineState,
    naive_bf16: ComputePipelineState,
    two_pass_2_d128: ComputePipelineState,
}

/// Buffer `[q_heads]` f32 des sinks d'une couche et kernels associés.
#[derive(Clone, Debug)]
struct AttentionSinks {
    buffer: Buffer,
    kernels: SinkKernels,
}

impl FullAttentionMetalState {
    /// Attache les sinks appris `[q_heads]` (f32) de la couche : l'attention
    /// decode passe alors par les kernels avec sink (naïf fenêtré ou 2-passes
    /// d128), jamais par le flash. Refusé sur un KV quantifié, dont le 2-passes
    /// n'a pas de variante avec sink.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le KV est quantifié, si le buffer est trop court ou
    /// si la compilation des kernels échoue.
    pub(crate) fn set_sinks(&mut self, arena: &DecodeResidentState, buffer: Buffer) -> Result<()> {
        if self.quant.is_some() {
            return Err(InferError::Config(format!(
                "sinks d'attention sur KV {:?} non supportés",
                self.format
            )));
        }
        let needed = self
            .q_heads
            .checked_mul(std::mem::size_of::<f32>())
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| InferError::Dimension("sinks: taille déborde".to_string()))?;
        if buffer.length() < needed {
            return Err(InferError::Dimension(format!(
                "sinks: buffer de {} octets, attendu {needed}",
                buffer.length()
            )));
        }
        // Le setup a déjà replié ce KV en dense : signalé une fois, comme le
        // repli `KV_QUANT` des couches sans attention quantifiée.
        if crate::runtime_flags::kv_quant_format().is_quantized() {
            static WARNED: std::sync::Once = std::sync::Once::new();
            WARNED.call_once(|| {
                eprintln!("SARAGOSSA_RUST_KV_QUANT ignoré (couches à sinks) : KV résident dense");
            });
        }
        self.sinks = Some(AttentionSinks {
            buffer,
            kernels: arena.sink_kernels()?.clone(),
        });
        Ok(())
    }

    /// Indique si la couche porte des sinks (chemins en lignes refusés).
    pub(super) fn has_sinks(&self) -> bool {
        self.sinks.is_some()
    }

    /// Renvoie la dimension d'une ligne KV (`kv_heads * head_dim`).
    pub(crate) fn kv_dim(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// Renvoie le nombre de lignes valides (= position courante), lignes
    /// absentes devant le buffer comprises.
    pub(crate) fn len(&self) -> usize {
        self.base + self.len
    }

    /// Lignes écrites dans le buffer, depuis sa ligne 0.
    pub(super) fn stored_len(&self) -> usize {
        self.len
    }

    /// Déclare `base` lignes absolues absentes devant la ligne 0 d'une arène
    /// encore vide : le semis suivant porte les lignes `[base, n)` (couche
    /// glissante bornée à sa fenêtre). Les kernels lisent le buffer comme
    /// avant ; seule la traduction des positions absolues change.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si l'arène porte déjà des lignes ou un préfixe adopté.
    pub(crate) fn rebase(&mut self, base: usize) -> Result<()> {
        if self.len != 0 || self.floor != 0 || self.adopted.is_some() {
            return Err(InferError::Dimension(format!(
                "base KV {base} sur une arène de {} lignes (plancher {})",
                self.len, self.floor
            )));
        }
        self.base = base;
        Ok(())
    }

    /// Ligne du buffer de la ligne absolue `row`.
    fn stored_row(&self, row: usize, label: &str) -> Result<usize> {
        row.checked_sub(self.base).ok_or_else(|| {
            InferError::Dimension(format!(
                "{label}: ligne {row} sortie du K/V borné (base {})",
                self.base
            ))
        })
    }

    /// Tronque le nombre de lignes K/V valides sans recopier les buffers.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `len` dépasse la capacité allouée ou descend
    /// sous le préfixe adopté ou la base.
    pub(crate) fn truncate(&mut self, len: usize) -> Result<()> {
        let stored = self.stored_row(len, "truncate KV")?;
        if stored > self.capacity || stored < self.floor {
            return Err(InferError::Dimension(format!(
                "truncate KV {len} hors [{}, {}] (base, préfixe adopté, capacité)",
                self.base + self.floor,
                self.base + self.capacity
            )));
        }
        self.len = stored;
        Ok(())
    }

    /// Renvoie la capacité en lignes.
    pub(super) fn scratch(&self) -> &ScratchPool {
        &self.scratch
    }

    pub(super) fn attention_2pass_2_d128(&self) -> &ComputePipelineState {
        &self.attention_2pass_2_d128
    }

    pub(super) fn attention_2pass_2(&self) -> &ComputePipelineState {
        &self.attention_2pass_2
    }

    /// Avance `len` de `rows` lignes appendues par un kernel en lignes.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la capacité serait dépassée.
    pub(super) fn advance_len(&mut self, rows: usize) -> Result<()> {
        let next = self
            .len
            .checked_add(rows)
            .filter(|next| *next <= self.capacity);
        match next {
            Some(next) => {
                self.len = next;
                Ok(())
            }
            None => Err(InferError::Dimension(format!(
                "append KV résident: {} + {rows} lignes dépasse la capacité {}",
                self.len, self.capacity
            ))),
        }
    }

    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// Relit `rows` lignes K/V (rope'd, f32) à partir de la ligne `from`, pour
    /// rapatrier vers le cache CPU les lignes ajoutées par un kernel en lignes.
    /// Un KV quantifié rend ses valeurs déquantifiées ; un ré-semis les
    /// requantifie avec les mêmes codes, l'échelle pouvant varier d'un ulp.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `from + rows` dépasse les lignes valides.
    pub(crate) fn read_rows(&self, from: usize, rows: usize) -> Result<(Vec<f32>, Vec<f32>)> {
        let from = self.stored_row(from, "read KV rows")?;
        let end = from
            .checked_add(rows)
            .filter(|end| *end <= self.len)
            .ok_or_else(|| {
                InferError::Dimension(format!(
                    "read KV rows: {from} + {rows} hors des {} lignes valides",
                    self.len
                ))
            })?;
        let kv_dim = self.kv_dim();
        let offset = from
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("read KV rows: offset déborde".to_string()))?;
        let count = (end - from)
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("read KV rows: longueur déborde".to_string()))?;
        if let Some(quant) = &self.quant {
            return Ok((
                quant.keys.read_rows(&self.keys, from, end - from, kv_dim)?,
                quant
                    .values
                    .read_rows(&self.values, from, end - from, kv_dim)?,
            ));
        }
        Ok((
            read_f32_at(&self.keys, offset, count)?,
            read_f32_at(&self.values, offset, count)?,
        ))
    }

    /// Renvoie le format du KV (dense ou quantifié) : seule source pour
    /// choisir un chemin selon le stockage, `keys().element()` ne distinguant
    /// pas Q8 de Q4.
    pub(crate) fn format(&self) -> KvFormat {
        self.format
    }

    /// Métadonnées du KV quantifié, `None` pour un KV dense.
    pub(super) fn quant(&self) -> Option<&KvQuant> {
        self.quant.as_ref()
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
        self.seed_segments(
            &[KvSegment::Cpu(keys_rows)],
            &[KvSegment::Cpu(values_rows)],
            n_rows,
        )
    }

    /// Comme [`Self::seed`], depuis des segments de lignes contiguës mis bout à
    /// bout (KV CPU segmenté) : aucun tampon intermédiaire de la taille du KV.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `n_rows > capacity` ou si les segments ne totalisent
    /// pas `n_rows * kv_dim`.
    pub(crate) fn seed_segments(
        &mut self,
        keys_segments: &[KvSegment<'_>],
        values_segments: &[KvSegment<'_>],
        n_rows: usize,
    ) -> Result<()> {
        // Arène adoptée : les segments suivent le préfixe déjà en place, à
        // partir du plancher, jamais réécrit. Arène bornée : ils commencent
        // à la base.
        let first = self.floor;
        let n_rows = self.stored_row(n_rows, "seed KV")?;
        if n_rows > self.capacity || first > n_rows {
            return Err(InferError::Dimension(format!(
                "seed KV {first}..{n_rows} hors [{}, {}]",
                self.floor, self.capacity
            )));
        }
        let kv_dim = self.kv_dim();
        let expected = (n_rows - first)
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("seed KV déborde".to_string()))?;
        let keys_len: usize = keys_segments.iter().map(KvSegment::len).sum();
        let values_len: usize = values_segments.iter().map(KvSegment::len).sum();
        if keys_len != expected || values_len != expected {
            return Err(InferError::Dimension(format!(
                "seed KV: attendu {expected} (={n_rows}×{kv_dim}), reçu keys={keys_len} values={values_len}"
            )));
        }
        if let Some(quant) = &self.quant {
            if first > 0 {
                return Err(InferError::Dimension(
                    "seed KV quantifié après un préfixe adopté".to_string(),
                ));
            }
            self.seed_quantized(quant, keys_segments, values_segments)?;
            self.len = n_rows;
            return Ok(());
        }
        // Diagnostic C1B : arrondi bf16 du seed sur K seul ou V seul, buffers f32.
        let (round_keys, round_values) = match self.keys.element() {
            GpuElement::F32 => (kv_bf16_sim_konly(), kv_bf16_sim_vonly()),
            _ => (false, false),
        };
        // Les vues GPU se copient sur le GPU : une écriture CPU dans l'arène
        // neuve payait une faute de page par 16 Kio (2,2 s à 256k).
        let mut gpu_copies = Vec::new();
        for (tensor, segments, round) in [
            (&self.keys, keys_segments, round_keys),
            (&self.values, values_segments, round_values && !round_keys),
        ] {
            let mut offset = first * kv_dim;
            for segment in segments {
                // Un segment bf16 sème une arène bf16 à l'identique ; une arène
                // f32 (queue courte, DFlash) le reçoit élargi : valeurs arrondies.
                match (segment.shared(), tensor.element()) {
                    (Some((buffer, byte_offset, input)), GpuElement::F32 | GpuElement::Bf16)
                        if !round =>
                    {
                        gpu_copies.push((
                            buffer,
                            byte_offset,
                            input,
                            tensor,
                            offset,
                            segment.len(),
                        ));
                    }
                    (_, GpuElement::F32) if round => {
                        let rounded: Vec<f32> = segment
                            .as_f32()
                            .iter()
                            .copied()
                            .map(bf16_round_f32)
                            .collect();
                        write_f32_at(tensor, offset, &rounded)?;
                    }
                    (_, GpuElement::F32) => write_f32_at(tensor, offset, &segment.as_f32())?,
                    (_, GpuElement::Bf16) if matches!(segment, KvSegment::CpuBf16(_)) => {
                        if let KvSegment::CpuBf16(bits) = segment {
                            crate::metal_backend::write_u16_buffer_at(
                                tensor.buffer(),
                                offset,
                                bits,
                            )?;
                        }
                    }
                    (_, GpuElement::Bf16) => {
                        write_f32_as_bf16_at(tensor, offset, &segment.as_f32())?;
                    }
                    (_, element @ (GpuElement::U32 | GpuElement::U8)) => {
                        return Err(InferError::Metal(format!(
                            "seed KV sur un buffer {element:?} invalide"
                        )));
                    }
                }
                offset += segment.len();
            }
        }
        // Command buffer créé seulement s'il sera commité (cf. `seed_quantized`) :
        // segments tous CPU (queue d'une arène adoptée, repli, diagnostic
        // d'arrondi), rien à soumettre.
        if !gpu_copies.is_empty() {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let encoder_guard = EncoderEndGuard::new(encoder);
            for (buffer, byte_offset, input, tensor, offset, len) in gpu_copies {
                self.encode_copy_at(
                    encoder,
                    buffer,
                    byte_offset,
                    input,
                    tensor.buffer(),
                    tensor.element(),
                    offset,
                    len,
                )?;
            }
            encoder_guard.end();
            commit_and_wait(command_buffer)?;
        }
        self.len = n_rows;
        Ok(())
    }

    /// Semis d'un KV quantifié : segments GPU quantifiés par le kernel,
    /// segments CPU par l'oracle CPU (mêmes octets). Chaque segment doit couvrir
    /// des lignes entières, la quantification groupant le long d'une ligne.
    fn seed_quantized(
        &self,
        quant: &KvQuant,
        keys_segments: &[KvSegment<'_>],
        values_segments: &[KvSegment<'_>],
    ) -> Result<()> {
        let kv_dim = self.kv_dim();
        let mut gpu_segments = Vec::new();
        for (payload, tensor, segments) in [
            (&self.keys, &quant.keys, keys_segments),
            (&self.values, &quant.values, values_segments),
        ] {
            let mut row = 0;
            for segment in segments {
                if segment.len() % kv_dim != 0 {
                    return Err(InferError::Dimension(format!(
                        "seed KV quantifié: segment de {} valeurs hors lignes de {kv_dim}",
                        segment.len()
                    )));
                }
                let rows = segment.len() / kv_dim;
                // Le kernel de quantification lit du f32 : un segment bf16 passe
                // par l'oracle CPU, élargi.
                match segment {
                    KvSegment::Shared(shared) if rows > 0 => {
                        gpu_segments.push((payload, tensor, *shared, row, rows));
                    }
                    KvSegment::Shared(_) => {}
                    KvSegment::Cpu(_) | KvSegment::CpuBf16(_) | KvSegment::SharedBf16(_) => {
                        tensor.write_rows(payload, row, kv_dim, &segment.as_f32())?;
                    }
                }
                row += rows;
            }
        }
        // Command buffer créé seulement s'il sera commité : un buffer jamais
        // commité reste compté par la file (64 au plus) faute de pool
        // d'autorelease, et bloque `new_command_buffer` après quelques dizaines
        // de semis.
        if gpu_segments.is_empty() {
            return Ok(());
        }
        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        for (payload, tensor, shared, row, rows) in gpu_segments {
            tensor.encode_rows(
                encoder,
                payload,
                shared.buffer(),
                shared.byte_offset(),
                kv_dim,
                row,
                rows,
                kv_dim,
            )?;
        }
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
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
        if let Some(quant) = &self.quant {
            quant.keys.write_rows(&self.keys, self.len, kv_dim, key)?;
            quant
                .values
                .write_rows(&self.values, self.len, kv_dim, value)?;
            self.len += 1;
            return Ok(());
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
            element @ (GpuElement::U32 | GpuElement::U8) => {
                return Err(InferError::Metal(format!(
                    "append KV sur un buffer {element:?} invalide"
                )));
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
        self.attention_decode_with(q, |encoder, q_buf, scores_buf, out_buf| {
            self.encode_attention_decode_windowed(encoder, q_buf, scores_buf, out_buf, window_start)
        })
    }

    /// Oracle GPU du 2-passes fenêtré avec sink : force le naïf avec sink sur
    /// les lignes `window_start..len`, quel que soit le choix de production.
    #[cfg(test)]
    pub(crate) fn attention_decode_sink_naive(
        &self,
        q: &[f32],
        window_start: usize,
    ) -> Result<Vec<f32>> {
        let sinks = self
            .sinks
            .as_ref()
            .ok_or_else(|| InferError::Config("naïf avec sink sans sinks".to_string()))?;
        let window_start = self.stored_row(window_start, "naïf avec sink")?;
        let dims = U32x4Param::Bytes([
            to_u32(self.q_heads, "q_heads")?,
            to_u32(self.kv_heads, "kv_heads")?,
            to_u32(self.head_dim, "head_dim")?,
            to_u32(self.len, "len")?,
        ]);
        self.attention_decode_with(q, |encoder, q_buf, scores_buf, out_buf| {
            self.encode_attention_decode_sink_naive(
                encoder,
                q_buf,
                scores_buf,
                out_buf,
                window_start,
                dims,
                sinks,
            )
        })
    }

    /// Force le 2-passes fenêtré avec sink, indépendamment des flags runtime
    /// (`SARAGOSSA_RUST_SDPA_2PASS=0` ferait retomber la production sur le naïf).
    #[cfg(test)]
    pub(crate) fn attention_decode_sink_two_pass(
        &self,
        q: &[f32],
        window_start: usize,
    ) -> Result<Vec<f32>> {
        let sinks = self
            .sinks
            .as_ref()
            .ok_or_else(|| InferError::Config("2-passes avec sink sans sinks".to_string()))?;
        let window_start = self.stored_row(window_start, "2-passes avec sink")?;
        self.attention_decode_with(q, |encoder, q_buf, _scores, out_buf| {
            self.encode_attention_decode_sink_windowed_2pass(
                encoder,
                q_buf,
                out_buf,
                window_start,
                self.len,
                sinks,
            )
        })
    }

    /// Exécute un encodage d'attention decode sur `q` uploadé et relit la sortie.
    fn attention_decode_with(
        &self,
        q: &[f32],
        encode: impl FnOnce(&ComputeCommandEncoderRef, &BufferRef, &BufferRef, &BufferRef) -> Result<()>,
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
        encode(
            encoder,
            q_buf.tensor().buffer(),
            scores.tensor().buffer(),
            out.tensor().buffer(),
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
        // Fenêtre absolue → ligne du buffer : une arène bornée relit les
        // mêmes lignes, dans le même ordre, depuis un pointeur décalé.
        let window_start = self.stored_row(window_start, "attention_decode fenêtre")?;
        if window_start >= self.len {
            return Err(InferError::Dimension(format!(
                "attention_decode fenêtre vide: début={window_start}, len={}",
                self.len
            )));
        }
        let len = to_u32(self.len, "len")?;
        let sdpa_dims = U32x4Param::Bytes([
            to_u32(self.q_heads, "q_heads")?,
            to_u32(self.kv_heads, "kv_heads")?,
            to_u32(self.head_dim, "head_dim")?,
            len,
        ]);
        let two_pass_dims = U32x4Param::Bytes([
            to_u32(self.q_heads, "2pass q_heads")?,
            to_u32(self.kv_heads, "2pass kv_heads")?,
            len,
            to_u32(sdpa_2pass_blocks(self.len), "2pass blocks")?,
        ]);
        self.encode_attention_decode_dims(
            encoder,
            q_buf,
            scores_buf,
            out_buf,
            window_start,
            self.len,
            sdpa_dims,
            two_pass_dims,
        )
    }

    /// Attention decode dont la longueur est lue par le GPU dans `ctrl` :
    /// ligne `row` sur `ctrl[slot] + row + 1` lignes. Mêmes kernels et même choix
    /// flash/2-passes, fixé par la borne hôte `max_position + row + 1`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde ou si l'allocation 2-passes échoue.
    pub(crate) fn encode_attention_decode_gpu_position(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
        rows: &GpuPositionRows<'_>,
        row: usize,
    ) -> Result<()> {
        // Ce chemin (bloc MTP) suppose une fenêtre nulle et n'a pas de variante
        // avec sink : refus explicite plutôt qu'une attention fausse. La
        // position GPU est absolue : une arène bornée est refusée aussi.
        if self.sinks.is_some() || self.base != 0 {
            return Err(InferError::Config(
                "attention en position GPU avec sinks ou fenêtre non supportée".to_string(),
            ));
        }
        let len_bound = rows
            .position()
            .max_position
            .checked_add(row + 1)
            .ok_or_else(|| InferError::Dimension("position GPU len déborde".to_string()))?;
        self.encode_attention_decode_dims(
            encoder,
            q_buf,
            scores_buf,
            out_buf,
            0,
            len_bound,
            rows.sdpa(row),
            rows.sdpa_2pass(row),
        )
    }

    /// Cœur SDPA decode. `len_bound` (= len hôte, ou borne en position GPU) choisit
    /// le kernel et la grille 2-passes ; `sdpa_dims`/`two_pass_dims` portent la
    /// longueur effective lue par le kernel.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers + fenêtre + borne + deux jeux de dims"
    )]
    fn encode_attention_decode_dims(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
        window_start: usize,
        len_bound: usize,
        sdpa_dims: U32x4Param<'_>,
        two_pass_dims: U32x4Param<'_>,
    ) -> Result<()> {
        if let Some(sinks) = &self.sinks {
            return self.encode_attention_decode_sink(
                encoder,
                q_buf,
                scores_buf,
                out_buf,
                window_start,
                len_bound,
                (sdpa_dims, two_pass_dims),
                sinks,
            );
        }
        // KV quantifié : 2-passes quelle que soit la longueur, seul kernel qui
        // le lit (aucune variante flash ni naïve). Les fenêtres glissantes sont
        // exclues dès la construction (`full_attention_dense`).
        if self.quant.is_some() {
            if window_start != 0 {
                return Err(InferError::Config(format!(
                    "attention_decode KV {:?}: fenêtre glissante non supportée",
                    self.format
                )));
            }
            return self.encode_attention_decode_2pass(
                encoder,
                q_buf,
                out_buf,
                len_bound,
                two_pass_dims,
                None,
                0,
            );
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
            && len_bound >= sdpa_2pass_min_len()
        {
            return self.encode_attention_decode_2pass(
                encoder,
                q_buf,
                out_buf,
                len_bound,
                two_pass_dims,
                None,
                0,
            );
        }
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
            (element @ (GpuElement::U32 | GpuElement::U8), _, _) => {
                return Err(InferError::Metal(format!(
                    "attention_decode sur KV {element:?} invalide"
                )));
            }
        };
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(self.keys.buffer()), 0);
        encoder.set_buffer(2, Some(self.values.buffer()), 0);
        if use_flash {
            encoder.set_buffer(3, Some(out_buf), 0);
            sdpa_dims.bind(encoder, 4);
        } else {
            encoder.set_buffer(3, Some(scores_buf), 0);
            encoder.set_buffer(4, Some(out_buf), 0);
            sdpa_dims.bind(encoder, 5);
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
            MTLSize::new(u64::from(to_u32(self.q_heads, "q_heads")?), 1, 1),
            MTLSize::new(threads, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Attention decode d'une couche à sinks. d128 et GQA réelle : 2-passes avec
    /// le sink en passe 2 (quelle que soit la longueur, les blocs vides étant
    /// neutralisés par la passe 2). Une fenêtre glissante devient une vue du KV
    /// décalée de `window_start` lignes. Sinon naïf fenêtré avec sink.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers + fenêtre + borne + dims + sinks"
    )]
    fn encode_attention_decode_sink(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
        window_start: usize,
        len_bound: usize,
        (sdpa_dims, two_pass_dims): (U32x4Param<'_>, U32x4Param<'_>),
        sinks: &AttentionSinks,
    ) -> Result<()> {
        // La passe 1 lance `32 × gqa` threads par threadgroup : borne par le
        // pipeline, sinon naïf (comme une GQA non divisible).
        let gqa = self.q_heads / self.kv_heads.max(1);
        let pass1_threads = match self.keys.element() {
            GpuElement::Bf16 => &self.attention_2pass_1_d128_bf16,
            _ => &self.attention_2pass_1_d128,
        }
        .max_total_threads_per_threadgroup();
        let two_pass = sdpa_2pass_enabled()
            && self.head_dim == 128
            && gqa > 1
            && self.q_heads % self.kv_heads == 0
            && u64::try_from(32 * gqa).is_ok_and(|threads| threads <= pass1_threads);
        if two_pass && window_start == 0 {
            return self.encode_attention_decode_2pass(
                encoder,
                q_buf,
                out_buf,
                len_bound,
                two_pass_dims,
                Some(sinks),
                0,
            );
        }
        if two_pass {
            return self.encode_attention_decode_sink_windowed_2pass(
                encoder,
                q_buf,
                out_buf,
                window_start,
                len_bound,
                sinks,
            );
        }
        self.encode_attention_decode_sink_naive(
            encoder,
            q_buf,
            scores_buf,
            out_buf,
            window_start,
            sdpa_dims,
            sinks,
        )
    }

    /// 2-passes avec sink sur la vue du KV commençant à `window_start`, sur la
    /// seule longueur de la fenêtre. Les blocs sont dimensionnés sur cette
    /// longueur (32 pour 513 lignes) : les dimensionner sur le contexte complet
    /// gaspillerait des partitions vides. Une fenêtre non nulle ne vient que de
    /// `encode_attention_decode_windowed`, dont la borne est la longueur hôte :
    /// les dims peuvent être reconstruites ici.
    fn encode_attention_decode_sink_windowed_2pass(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        out_buf: &BufferRef,
        window_start: usize,
        len_bound: usize,
        sinks: &AttentionSinks,
    ) -> Result<()> {
        if len_bound != self.len {
            return Err(InferError::Config(format!(
                "attention_decode 2pass fenêtré: borne {len_bound} ≠ len hôte {}",
                self.len
            )));
        }
        let window_len = len_bound.checked_sub(window_start).ok_or_else(|| {
            InferError::Dimension(format!(
                "attention_decode fenêtre vide: début={window_start}, borne={len_bound}"
            ))
        })?;
        let window_dims = U32x4Param::Bytes([
            to_u32(self.q_heads, "2pass q_heads")?,
            to_u32(self.kv_heads, "2pass kv_heads")?,
            to_u32(window_len, "2pass fenêtre")?,
            to_u32(sdpa_2pass_blocks(window_len), "2pass blocks")?,
        ]);
        self.encode_attention_decode_2pass(
            encoder,
            q_buf,
            out_buf,
            window_len,
            window_dims,
            Some(sinks),
            window_start,
        )
    }

    /// Naïf avec sink sur les lignes `window_start..len` : un threadgroup par
    /// tête Q, softmax sériel. Repli hors d128/GQA et oracle du 2-passes.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers + fenêtre + dims + sinks"
    )]
    fn encode_attention_decode_sink_naive(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        scores_buf: &BufferRef,
        out_buf: &BufferRef,
        window_start: usize,
        sdpa_dims: U32x4Param<'_>,
        sinks: &AttentionSinks,
    ) -> Result<()> {
        let pipeline = match self.keys.element() {
            GpuElement::F32 => &sinks.kernels.naive_f32,
            GpuElement::Bf16 => &sinks.kernels.naive_bf16,
            element @ (GpuElement::U32 | GpuElement::U8) => {
                return Err(InferError::Metal(format!(
                    "attention_decode sink sur KV {element:?} invalide"
                )));
            }
        };
        let window_start = u32::try_from(window_start)
            .map_err(|_| InferError::Dimension("window_start hors u32".to_string()))?;
        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(self.keys.buffer()), 0);
        encoder.set_buffer(2, Some(self.values.buffer()), 0);
        encoder.set_buffer(3, Some(scores_buf), 0);
        encoder.set_buffer(4, Some(out_buf), 0);
        sdpa_dims.bind(encoder, 5);
        encoder.set_bytes(
            6,
            std::mem::size_of::<u32>() as u64,
            std::ptr::from_ref(&window_start).cast::<c_void>(),
        );
        encoder.set_buffer(7, Some(&sinks.buffer), 0);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(to_u32(self.q_heads, "q_heads")?), 1, 1),
            MTLSize::new(256, 1, 1),
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
    ///
    /// `sinks = Some` remplace la passe 2 d128 par sa variante avec sink (passe 1
    /// inchangée : le sink ne touche aucune partition). `first_row` décale la vue
    /// K/V lue par la passe 1 (début de fenêtre glissante, 0 sinon).
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers + borne + dims + sinks + début de vue KV"
    )]
    fn encode_attention_decode_2pass(
        &self,
        encoder: &ComputeCommandEncoderRef,
        q_buf: &BufferRef,
        out_buf: &BufferRef,
        len_bound: usize,
        dims: U32x4Param<'_>,
        sinks: Option<&AttentionSinks>,
        first_row: usize,
    ) -> Result<()> {
        // Grille dimensionnée par la borne : en position GPU, les blocs au-delà de
        // la longueur effective restent vides (max=-inf, somme 0) et la passe 2
        // les neutralise (facteur exp(-inf)=0), comme les simdgroups vides du flash.
        let blocks = sdpa_2pass_blocks(len_bound);
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

        let kv_heads = to_u32(self.kv_heads, "2pass kv_heads")?;
        let blocks_u32 = to_u32(blocks, "2pass blocks")?;
        // Passe 1 : partiels par (kv_head, bloc).
        let pass1 = match &self.quant {
            Some(quant) => &quant.attention,
            None => match (self.keys.element(), self.head_dim) {
                (GpuElement::F32, 256) => &self.attention_2pass_1,
                (GpuElement::F32, 128) => &self.attention_2pass_1_d128,
                (GpuElement::Bf16, 256) => &self.attention_2pass_1_bf16,
                (GpuElement::Bf16, 128) => &self.attention_2pass_1_d128_bf16,
                (GpuElement::F32 | GpuElement::Bf16, other) => {
                    return Err(InferError::Metal(format!(
                        "attention_decode 2pass head_dim={other} non supporté"
                    )));
                }
                (element @ (GpuElement::U32 | GpuElement::U8), _) => {
                    return Err(InferError::Metal(format!(
                        "attention_decode 2pass sur KV {element:?} invalide"
                    )));
                }
            },
        };
        // NOTE: Le KV quantifié porte des échelles indexées par ligne absolue et
        // ne peut pas être décalé. `first_row != 0` ne vient que d'une couche à
        // sinks, que `set_sinks` refuse dès la construction sur KV quantifié.
        let kv_offset = first_row
            .checked_mul(self.kv_dim())
            .and_then(|cells| cells.checked_mul(self.keys.element().byte_size()))
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| InferError::Dimension("2pass offset fenêtre déborde".to_string()))?;
        encoder.set_compute_pipeline_state(pass1);
        encoder.set_buffer(0, Some(q_buf), 0);
        encoder.set_buffer(1, Some(self.keys.buffer()), kv_offset);
        encoder.set_buffer(2, Some(self.values.buffer()), kv_offset);
        encoder.set_buffer(3, Some(partials.tensor().buffer()), 0);
        encoder.set_buffer(4, Some(sums.tensor().buffer()), 0);
        encoder.set_buffer(5, Some(maxs.tensor().buffer()), 0);
        dims.bind(encoder, 6);
        if let Some(quant) = &self.quant {
            encoder.set_buffer(7, Some(quant.keys.scales().buffer()), 0);
            encoder.set_buffer(8, Some(quant.values.scales().buffer()), 0);
        }
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(kv_heads), u64::from(blocks_u32), 1),
            MTLSize::new(32, u64::try_from(gqa).unwrap_or(1), 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);

        // Passe 2 : réduction des blocs par q_head.
        let pass2 = match (self.head_dim, sinks) {
            (128, Some(sinks)) => &sinks.kernels.two_pass_2_d128,
            (other, Some(_)) => {
                return Err(InferError::Metal(format!(
                    "attention_decode 2pass avec sink head_dim={other} non supporté"
                )));
            }
            (256, None) => &self.attention_2pass_2,
            (128, None) => &self.attention_2pass_2_d128,
            (other, None) => {
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
        if let Some(sinks) = sinks {
            encoder.set_buffer(5, Some(&sinks.buffer), 0);
        }
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(u64::from(to_u32(self.q_heads, "2pass q_heads")?), 1, 1),
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
        if let Some(quant) = &self.quant {
            for (tensor, payload, input, offset) in [
                (&quant.keys, &self.keys, k_buf, k_offset),
                (&quant.values, &self.values, v_buf, v_offset),
            ] {
                tensor.encode_rows(encoder, payload, input, offset, kv_dim, self.len, 1, kv_dim)?;
            }
            self.len += 1;
            return Ok(());
        }
        let offset = self.len * kv_dim;
        self.encode_copy_at(
            encoder,
            k_buf,
            k_offset,
            GpuElement::F32,
            self.keys.buffer(),
            self.keys.element(),
            offset,
            kv_dim,
        )?;
        self.encode_copy_at(
            encoder,
            v_buf,
            v_offset,
            GpuElement::F32,
            self.values.buffer(),
            self.values.element(),
            offset,
            kv_dim,
        )?;
        self.len += 1;
        Ok(())
    }

    /// Append K/V à la position GPU `min(ctrl[slot], max_position) + row`, sans
    /// lire ni modifier `len` hôte (resync par l'appelant après le bloc).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le KV n'est pas f32 ou si `max_position + row`
    /// dépasse la capacité (vérifié avant dispatch).
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: buffers K/V + offsets + position GPU + ligne"
    )]
    pub(crate) fn encode_append_kv_gpu_position(
        &self,
        encoder: &ComputeCommandEncoderRef,
        k_buf: &BufferRef,
        k_offset: u64,
        v_buf: &BufferRef,
        v_offset: u64,
        rows: &GpuPositionRows<'_>,
        row: usize,
    ) -> Result<()> {
        if self.base != 0 {
            return Err(InferError::Config(
                "append KV en position GPU sur une arène bornée".to_string(),
            ));
        }
        let position = rows.position();
        let pipeline = rows.copy_at_position();
        check_gpu_position_rows(
            self.capacity,
            self.kv_dim(),
            self.format,
            0,
            position.max_position,
            row + 1,
        )?;
        let p = [
            to_u32(self.kv_dim(), "append position kv_dim")?,
            to_u32(position.slot, "append position slot")?,
            to_u32(row, "append position row")?,
            to_u32(position.max_position, "append position max")?,
        ];
        for (input, offset, output) in [
            (k_buf, k_offset, self.keys.buffer()),
            (v_buf, v_offset, self.values.buffer()),
        ] {
            encoder.set_compute_pipeline_state(pipeline);
            encoder.set_buffer(0, Some(input), offset);
            encoder.set_buffer(1, Some(output), 0);
            encoder.set_bytes(
                2,
                std::mem::size_of::<[u32; 4]>() as u64,
                p.as_ptr().cast::<c_void>(),
            );
            encoder.set_buffer(3, Some(position.ctrl), 0);
            let width = pipeline.thread_execution_width().max(1);
            crate::metal_backend::profile_dispatch();
            encoder.dispatch_threads(
                MTLSize::new(u64::from(p[0]), 1, 1),
                MTLSize::new(width, 1, 1),
            );
            crate::metal_backend::post_dispatch_barrier(encoder);
        }
        Ok(())
    }

    /// Copie `input[0..n]` → `output[offset..offset+n]` (device, `output` lié à un
    /// offset en octets). Brique de l'append KV résident.
    #[allow(
        clippy::too_many_arguments,
        reason = "primitive Metal chaude: buffers, formats et offsets reflètent les bindings du kernel"
    )]
    fn encode_copy_at(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        input_offset: u64,
        input_element: GpuElement,
        output: &BufferRef,
        output_element: GpuElement,
        output_offset: usize,
        n: usize,
    ) -> Result<()> {
        let len = u32::try_from(n)
            .map_err(|_| InferError::Dimension("copy_at n hors u32".to_string()))?;
        let offset_bytes = byte_offset(output_offset, output_element, "copy_at offset")?;
        let pipeline = match (input_element, output_element) {
            (GpuElement::F32, GpuElement::F32) => &self.copy_at,
            (GpuElement::F32, GpuElement::Bf16) => &self.copy_at_f32_to_bf16,
            (GpuElement::Bf16, GpuElement::Bf16) => &self.copy_at_bf16,
            (GpuElement::Bf16, GpuElement::F32) => &self.copy_at_bf16_to_f32,
            (input, output) => {
                return Err(InferError::Metal(format!(
                    "copy_at KV {input:?} vers un buffer {output:?} invalide"
                )));
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
