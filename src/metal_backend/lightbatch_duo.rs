//! Encodage duo (light-batch M=2, E2.2) : projections denses batchées en qmm2
//! (poids lus UNE fois pour les 2 flux), cœur conv/SSM et argmax PAR FLUX sur
//! l'état de chaque flux.
//!
//! Byte-identité : chaque composition dé-fusionnée utilisée ici est prouvée
//! BIT-exacte vs le chemin solo fusionné (oracles `qmm2_bitwise_*`,
//! `rms_simd_*`, `attn_gate_*` — commit E2.2a). Le routage qmm2 est GATÉ en
//! amont ([`MetalExecutor::qmm2_eligible_weight`]) : jamais de repli silencieux
//! vers le matmul générique (non bit-identique au qmv).

use super::*;
use std::cell::RefCell;

/// Une entrée de collecte : `([indices_flux_a, indices_flux_b], top_k)` pour
/// UNE couche MoE duo.
type ExpertIndicesPair = ([Buffer; 2], usize);

thread_local! {
    // Collecte diagnostique des indices d'experts du pas duo
    // (RETI_RUST_LIGHTBATCH_EXPERT_STATS) : une paire de buffers u32 [top_k]
    // par couche MoE duo, dans l'ordre des couches. None = collecte inactive.
    static EXPERT_INDICES_COLLECTOR: RefCell<Option<Vec<ExpertIndicesPair>>> =
        const { RefCell::new(None) };
}

/// Arme la collecte des indices d'experts pour le PROCHAIN pas duo encodé sur
/// ce thread (diagnostic disjonction M=2). À drainer via
/// [`take_expert_indices_collection`] après le wait du command buffer.
pub(crate) fn begin_expert_indices_collection() {
    EXPERT_INDICES_COLLECTOR.with(|slot| *slot.borrow_mut() = Some(Vec::new()));
}

/// Draine la collecte armée par [`begin_expert_indices_collection`] : une
/// entrée `([indices_flux_a, indices_flux_b], top_k)` par couche MoE duo.
pub(crate) fn take_expert_indices_collection() -> Option<Vec<([Buffer; 2], usize)>> {
    EXPERT_INDICES_COLLECTOR.with(|slot| slot.borrow_mut().take())
}

fn expert_indices_collection_active() -> bool {
    EXPERT_INDICES_COLLECTOR.with(|slot| slot.borrow().is_some())
}

fn push_expert_indices_pair(pair: [Buffer; 2], top_k: usize) {
    EXPERT_INDICES_COLLECTOR.with(|slot| {
        if let Some(collector) = slot.borrow_mut().as_mut() {
            collector.push((pair, top_k));
        }
    });
}

/// Paramètres de sampling d'UN flux du pas duo (E2.4) : miroir métal-côté de
/// `ResidentSampleSpec` (decoder), `rng_state` PAR FLUX.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DuoSampleParams {
    pub(crate) temperature: f32,
    pub(crate) top_p: f32,
    pub(crate) top_k: usize,
    pub(crate) rng_state: u64,
}

impl MetalExecutor {
    /// Renvoie `true` si `weight` route vers un kernel qmm2 à batch=2 (4-bit OU
    /// 8-bit gs64, `in_dim % 512 == 0`, `out_dim % 8 == 0`) — précondition du
    /// duo (jamais de repli silencieux vers le matmul générique).
    pub(crate) fn qmm2_eligible_weight(&self, weight: &MetalLinearWeightBuffers) -> bool {
        match weight {
            MetalLinearWeightBuffers::Dense { .. } => false,
            MetalLinearWeightBuffers::AffineQuantized {
                out_dim,
                in_dim,
                group_size,
                bits,
                ..
            } => {
                can_use_fast_affine_qmm2_buffers(2, *in_dim, *out_dim, *group_size, *bits)
                    || can_use_fast_affine_qmm2_u8_buffers(2, *in_dim, *out_dim, *group_size, *bits)
            }
        }
    }

    fn qmm2_eligible_weight_report(&self, name: &str, weight: &MetalLinearWeightBuffers) -> String {
        match weight {
            MetalLinearWeightBuffers::Dense {
                out_dim, in_dim, ..
            } => format!(
                "{name}=dense[{out_dim},{in_dim}] batch2={}",
                self.moe_router_duo_eligible(weight)
            ),
            MetalLinearWeightBuffers::AffineQuantized {
                out_dim,
                in_dim,
                group_size,
                bits,
                ..
            } => {
                let eligible = self.qmm2_eligible_weight(weight);
                format!("{name}=aq[{out_dim},{in_dim}] bits={bits} gs={group_size} qmm2={eligible}")
            }
        }
    }

    fn moe_router_duo_eligible(&self, weight: &MetalLinearWeightBuffers) -> bool {
        match weight {
            MetalLinearWeightBuffers::Dense { .. } => true,
            MetalLinearWeightBuffers::AffineQuantized { .. } => self.qmm2_eligible_weight(weight),
        }
    }

    /// Renvoie `true` si le chemin SOLO fusionne le prologue rms avec ce poids
    /// (kernels `affine_qmv_rms_fast`/`affine_qkv_split_rms_qmv_fast`, 4-bit
    /// gs64 uniquement). Le duo doit alors normaliser via `rms_norm_simd`
    /// (réduction bit-identique au prologue fusionné) ; sinon le solo passe par
    /// `rms_norm_rows` et le duo fait de même (rows=2, bit-identique par row).
    /// `epilogue` : `true` pour le site qkv (gate supplémentaire
    /// `fused_attn_epilogue`), `false` pour le site in_proj linear-attn.
    pub(crate) fn solo_rms_fusion_applies(
        &self,
        weight: &MetalLinearWeightBuffers,
        in_dim: usize,
        epilogue: bool,
    ) -> bool {
        if !fused_rms_prologue_enabled() || (epilogue && !fused_attn_epilogue_enabled()) {
            return false;
        }
        match weight {
            MetalLinearWeightBuffers::Dense { .. } => false,
            MetalLinearWeightBuffers::AffineQuantized {
                out_dim,
                in_dim: weight_in_dim,
                group_size,
                bits,
                ..
            } => {
                *weight_in_dim == in_dim
                    && ((fast_affine_qmv_enabled(*out_dim)
                        && *bits == FAST_QMV_BITS
                        && *group_size == FAST_QMV_GROUP_SIZE
                        && in_dim % 512 == 0)
                        || can_use_fast_affine_qmv_u8_buffers(
                            1,
                            in_dim,
                            *out_dim,
                            *group_size,
                            *bits,
                        ))
            }
        }
    }

    /// Encode la norm d'entrée duo en MIROIR du chemin solo : `rms_norm_simd`
    /// si le solo fusionne le prologue rms avec `weight`, sinon
    /// `rms_norm_rows` (rows=2, même kernel que le solo dé-fusionné).
    #[expect(
        clippy::too_many_arguments,
        reason = "wrapper d'encodage : buffers + dims + poids pilote du choix"
    )]
    pub(crate) fn encode_rms_norm_duo_matching_solo(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        weight_buffer: &BufferRef,
        output_buffer: &BufferRef,
        dim: usize,
        eps: f32,
        proj_weight: &MetalLinearWeightBuffers,
        epilogue: bool,
    ) -> Result<()> {
        if self.solo_rms_fusion_applies(proj_weight, dim, epilogue) {
            self.encode_rms_norm_simd_rows(
                encoder,
                input_buffer,
                weight_buffer,
                output_buffer,
                2,
                dim,
                eps,
            )
        } else {
            self.encode_rms_norm_rows(
                encoder,
                input_buffer,
                weight_buffer,
                output_buffer,
                2,
                dim,
                eps,
            )
        }
    }

    /// Renvoie `true` si les projections in/out de la linear-attn résidente
    /// routent vers qmm2 à batch=2 (précondition du duo).
    pub(crate) fn qmm2_eligible_linear_attn(
        &self,
        weights: &MetalLinearAttnResidentDenseWeights,
    ) -> bool {
        let Some(full) = weights.full.as_ref() else {
            return false;
        };
        self.qmm2_eligible_weight(&full.in_proj) && self.qmm2_eligible_weight(&full.out_proj)
    }

    /// Encode UNE couche linear-attn duo : `rms_simd` rows=2 → in_proj qmm2 →
    /// conv/gates/delta/rms_gate PAR FLUX (état de chaque flux, scratch
    /// namespacé par slot) → out_proj qmm2. `input_buffer`/`output_buffer` =
    /// `[2, in_dim]`/`[2, hidden]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incohérente ou si un encodage
    /// Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "data flow duo : 2 états + poids + dims + ping-pong"
    )]
    pub(crate) fn encode_linear_attn_resident_duo_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        input_norm: (&BufferRef, f32),
        output_buffer: &BufferRef,
        weights: &MetalLinearAttnResidentWeights,
        states: [&LinearAttentionMetalState; 2],
        slots: [u64; 2],
        spec: LinearAttentionStepSpec,
        dims: LinearAttnResidentDims,
    ) -> Result<()> {
        let LinearAttnResidentDims {
            in_dim,
            conv_dim,
            value_dim,
            key_dim,
        } = dims;
        let in_proj_dim = conv_dim
            .checked_add(value_dim)
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .and_then(|value| value.checked_add(spec.num_value_heads))
            .ok_or_else(|| InferError::Dimension("duo in_proj concat déborde".to_string()))?;
        let normed2 = self.private_f32_buffer(2 * in_dim, "lightbatch_la_normed2")?;
        let in_proj2 = self.private_f32_buffer(2 * in_proj_dim, "lightbatch_la_in_proj2")?;
        let gated2 = self.private_f32_buffer(2 * value_dim, "lightbatch_la_gated2")?;

        let (norm_weight, eps) = input_norm;
        self.encode_rms_norm_duo_matching_solo(
            encoder,
            input_buffer,
            norm_weight,
            &normed2,
            in_dim,
            eps,
            &weights.in_proj,
            false,
        )?;
        let projected = self.encode_matmul_weight_buffers(
            encoder,
            &normed2,
            2,
            in_dim,
            &weights.in_proj,
            &in_proj2,
            false,
        )?;
        if projected != in_proj_dim {
            return Err(InferError::Dimension(format!(
                "duo in_proj sort {projected}, attendu {in_proj_dim}"
            )));
        }

        for (stream, state) in states.into_iter().enumerate() {
            let _namespace = install_scratch_namespace(slots[stream]);
            let row_base = stream
                .checked_mul(in_proj_dim)
                .ok_or_else(|| InferError::Dimension("duo in_proj row déborde".to_string()))?;
            let row_offset = byte_offset_f32(row_base, "duo in_proj row offset")?;
            let z_offset = byte_offset_f32(
                row_base
                    .checked_add(conv_dim)
                    .ok_or_else(|| InferError::Dimension("duo z offset déborde".to_string()))?,
                "duo z offset",
            )?;
            let beta_offset = byte_offset_f32(
                row_base
                    .checked_add(conv_dim)
                    .and_then(|value| value.checked_add(value_dim))
                    .ok_or_else(|| InferError::Dimension("duo beta offset déborde".to_string()))?,
                "duo beta offset",
            )?;
            let gate_offset = byte_offset_f32(
                row_base
                    .checked_add(conv_dim)
                    .and_then(|value| value.checked_add(value_dim))
                    .and_then(|value| value.checked_add(spec.num_value_heads))
                    .ok_or_else(|| InferError::Dimension("duo gate offset déborde".to_string()))?,
                "duo gate offset",
            )?;

            // Mêmes labels que le chemin solo : le namespace par slot rend les
            // buffers DISJOINTS entre flux (slot 0 réutilise ceux du solo).
            let conv_out = self.private_f32_buffer(conv_dim, "linear_attn_conv_out")?;
            let q_norm = self.private_f32_buffer(key_dim, "linear_attn_q_norm")?;
            let k_norm = self.private_f32_buffer(key_dim, "linear_attn_k_norm")?;
            let beta = self.private_f32_buffer(spec.num_value_heads, "linear_attn_beta")?;
            let decay = self.private_f32_buffer(spec.num_value_heads, "linear_attn_decay")?;
            let y = self.private_f32_buffer(value_dim, "linear_attn_y")?;
            let gated = self.private_f32_buffer(value_dim, "linear_attn_gated")?;

            self.encode_linear_attn_conv_with_offset(
                encoder,
                &in_proj2,
                row_offset,
                &weights.conv_weight,
                &state.conv,
                &conv_out,
                conv_dim,
                spec.conv_kernel_dim,
            )?;
            self.encode_linear_attn_norm_gates_with_offsets(
                encoder,
                &conv_out,
                &in_proj2,
                beta_offset,
                &in_proj2,
                gate_offset,
                &weights.a_log,
                &weights.dt_bias,
                &q_norm,
                &k_norm,
                &beta,
                &decay,
                spec,
            )?;
            self.encode_linear_attn_gated_delta(
                encoder,
                &conv_out,
                &q_norm,
                &k_norm,
                &beta,
                &decay,
                &state.ssm,
                state.ssm_bf16,
                &y,
                spec,
            )?;
            self.encode_linear_attn_rms_gate_with_offset(
                encoder,
                &y,
                &in_proj2,
                z_offset,
                &weights.norm_weight,
                &gated,
                spec,
            )?;
            let gated_offset = byte_offset_f32(
                stream
                    .checked_mul(value_dim)
                    .ok_or_else(|| InferError::Dimension("duo gated row déborde".to_string()))?,
                "duo gated row offset",
            )?;
            self.encode_copy_with_offsets(encoder, &gated, 0, &gated2, gated_offset, value_dim)?;
        }

        let out_dim = self.encode_matmul_weight_buffers(
            encoder,
            &gated2,
            2,
            value_dim,
            &weights.out_proj,
            output_buffer,
            false,
        )?;
        if out_dim != in_dim {
            return Err(InferError::Dimension(format!(
                "duo out_proj sort {out_dim}, attendu {in_dim}"
            )));
        }
        Ok(())
    }

    /// Encode le lm_head + argmax duo : logits `[2, vocab]` par UN qmm2 (poids
    /// lm_head lus une fois) puis argmax PAR FLUX (mêmes kernels que le solo)
    /// vers `index_buffer[0..2]` (u32 par flux).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la projection sort une dimension inattendue ou si
    /// un encodage Metal échoue.
    pub(crate) fn encode_lm_head_argmax_duo_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        in_dim: usize,
        slots: [u64; 2],
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits2 = self.private_f32_buffer(2 * out_dim, "lightbatch_argmax_logits2")?;
        let projected = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            2,
            in_dim,
            lm_head,
            &logits2,
            true,
        )?;
        if projected != out_dim {
            return Err(InferError::Dimension(format!(
                "duo lm_head sort {projected}, attendu {out_dim}"
            )));
        }
        let partial_count = out_dim.div_ceil(256);
        for (stream, slot) in slots.into_iter().enumerate() {
            let _namespace = install_scratch_namespace(slot);
            let partial_values = self.private_f32_buffer(partial_count, "argmax_partial_values")?;
            let partial_indices =
                self.private_u32_buffer(partial_count, "argmax_partial_indices")?;
            let logits_offset = byte_offset_f32(
                stream
                    .checked_mul(out_dim)
                    .ok_or_else(|| InferError::Dimension("duo logits row déborde".to_string()))?,
                "duo logits row offset",
            )?;
            let index_offset = u64::try_from(stream * std::mem::size_of::<u32>())
                .map_err(|_| InferError::Metal("duo index offset hors u64".to_string()))?;
            self.encode_argmax_blocks_with_offset(
                encoder,
                &logits2,
                logits_offset,
                &partial_values,
                &partial_indices,
                out_dim,
            )?;
            self.encode_argmax_finalize_with_offset(
                encoder,
                &partial_values,
                &partial_indices,
                index_buffer,
                index_offset,
                partial_count,
            )?;
        }
        Ok(())
    }

    /// Encode le lm_head + sampler duo (E2.4) : logits `[2, vocab]` par UN qmm2
    /// (poids lm_head lus une fois) puis sampler on-device PAR FLUX (`top_k<=32`
    /// ou Gumbel `top_k=0/top_p=1`) vers `index_buffer[0..2]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si la projection sort une dimension inattendue, si un
    /// `top_k` est invalide ou si un encodage Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage sampler duo : buffers + params par flux + slots"
    )]
    pub(crate) fn encode_lm_head_sample_duo_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input_buffer: &BufferRef,
        lm_head: &MetalLinearWeightBuffers,
        index_buffer: &BufferRef,
        in_dim: usize,
        samples: [DuoSampleParams; 2],
        slots: [u64; 2],
    ) -> Result<()> {
        let out_dim = self.linear_weight_out_dim(lm_head);
        let logits2 = self.private_f32_buffer(2 * out_dim, "lightbatch_sample_logits2")?;
        let projected = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            2,
            in_dim,
            lm_head,
            &logits2,
            false,
        )?;
        if projected != out_dim {
            return Err(InferError::Dimension(format!(
                "duo lm_head sample sort {projected}, attendu {out_dim}"
            )));
        }
        for (stream, slot) in slots.into_iter().enumerate() {
            let _namespace = install_scratch_namespace(slot);
            let sample = samples[stream];
            let logits_offset = byte_offset_f32(
                stream
                    .checked_mul(out_dim)
                    .ok_or_else(|| InferError::Dimension("duo logits row déborde".to_string()))?,
                "duo sample logits offset",
            )?;
            let index_offset = u64::try_from(stream * std::mem::size_of::<u32>())
                .map_err(|_| InferError::Metal("duo sample index offset hors u64".to_string()))?;
            self.encode_sample_topk_topp_with_offsets(
                encoder,
                &logits2,
                logits_offset,
                index_buffer,
                index_offset,
                out_dim,
                sample.top_k,
                sample.temperature,
                sample.top_p,
                sample.rng_state,
            )?;
        }
        Ok(())
    }

    /// Renvoie `true` si le tail MoE duo s'applique : router batchable
    /// (dense ou qmm2 quantifié) et shared expert (gate/up/down) routable qmm2
    /// (E2.3). Les gathers routés restent par flux quoi qu'il arrive (experts
    /// disjoints → trafic incompressible).
    pub(crate) fn moe_shared_duo_eligible(&self, weights: &MetalMoeSharedWeights) -> bool {
        let eligible = self.moe_router_duo_eligible(&weights.router)
            && self.qmm2_eligible_weight(&weights.shared_gate_proj)
            && self.qmm2_eligible_weight(&weights.shared_up_proj)
            && self.qmm2_eligible_weight(&weights.shared_down_proj);
        if !eligible && crate::runtime_flags::env_flag("RETI_RUST_TRACE_MOE", false) {
            eprintln!(
                "MoE shared duo qmm2 eligibility: {}; {}; {}; {}",
                self.qmm2_eligible_weight_report("router", &weights.router),
                self.qmm2_eligible_weight_report("shared_gate_proj", &weights.shared_gate_proj),
                self.qmm2_eligible_weight_report("shared_up_proj", &weights.shared_up_proj),
                self.qmm2_eligible_weight_report("shared_down_proj", &weights.shared_down_proj)
            );
        }
        eligible
    }

    /// Encode le tail MoE shared duo (E2.3) : router + shared expert batchés en
    /// qmm2 (poids lus UNE fois pour les 2 flux), topk/gathers routés/sommes PAR
    /// FLUX (kernels du solo, scratch namespacé par slot). `input2`/`residual2`/
    /// `output2` = `[2, in_dim]`.
    ///
    /// Byte-identité : router/shared via qmm2 ≡ qmv par ligne (oracles E2.2a) ;
    /// le shared dé-fusionné (qmm2 gate + qmm2 up + swiglu élémentaire) ≡ le
    /// fusé `gate_up_swiglu_fast` du solo (oracle E2.3) ; topk, gathers,
    /// weighted_sum et add_sigmoid sont les kernels du solo sur des copies
    /// bit-exactes de la ligne du flux.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est incohérente ou si un encodage
    /// Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "tail MoE duo : duo buffers + poids + top_k + slots"
    )]
    pub(crate) fn encode_moe_shared_duo_buffers(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input2: &BufferRef,
        residual2: &BufferRef,
        output2: &BufferRef,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
        slots: [u64; 2],
    ) -> Result<()> {
        let shape = self.check_moe_shared_buffer_shapes(in_dim, weights, top_k)?;
        let expert_count = shape.expert_count;
        let inter_dim = shape.inter_dim;
        let out_dim = shape.out_dim;
        let shared_inter_dim = shape.shared_inter_dim;
        if out_dim != in_dim {
            return Err(InferError::Dimension(format!(
                "MoE duo attend out_dim == in_dim, reçu {out_dim} != {in_dim}"
            )));
        }

        // Parties batchées : router + shared expert, poids lus UNE fois.
        let router2 = self.private_f32_buffer(2 * expert_count, "lightbatch_moe_router2")?;
        let sgate2 = self.private_f32_buffer(2 * shared_inter_dim, "lightbatch_moe_sgate2")?;
        let sup2 = self.private_f32_buffer(2 * shared_inter_dim, "lightbatch_moe_sup2")?;
        let shidden2 = self.private_f32_buffer(2 * shared_inter_dim, "lightbatch_moe_shidden2")?;
        let sdown2 = self.private_f32_buffer(2 * out_dim, "lightbatch_moe_sdown2")?;
        let router_out = self.encode_matmul_weight_buffers(
            encoder,
            input2,
            2,
            in_dim,
            &weights.router,
            &router2,
            false,
        )?;
        if router_out != expert_count {
            return Err(InferError::Dimension(format!(
                "MoE duo router sort {router_out}, attendu {expert_count}"
            )));
        }
        let sgate_out = self.encode_matmul_weight_buffers(
            encoder,
            input2,
            2,
            in_dim,
            &weights.shared_gate_proj,
            &sgate2,
            false,
        )?;
        let sup_out = self.encode_matmul_weight_buffers(
            encoder,
            input2,
            2,
            in_dim,
            &weights.shared_up_proj,
            &sup2,
            false,
        )?;
        if sgate_out != shared_inter_dim || sup_out != shared_inter_dim {
            return Err(InferError::Dimension(format!(
                "MoE duo shared gate={sgate_out} up={sup_out}, attendu {shared_inter_dim}"
            )));
        }
        self.encode_swiglu(
            encoder,
            owned_buffers,
            &sgate2,
            &sup2,
            &shidden2,
            2 * shared_inter_dim,
        )?;
        let sdown_out = self.encode_matmul_weight_buffers(
            encoder,
            &shidden2,
            2,
            shared_inter_dim,
            &weights.shared_down_proj,
            &sdown2,
            false,
        )?;
        if sdown_out != out_dim {
            return Err(InferError::Dimension(format!(
                "MoE duo shared down sort {sdown_out}, attendu {out_dim}"
            )));
        }

        // Parties par flux : topk + shared gate scalaire + gathers routés +
        // sommes — composition du solo sur des copies de la ligne du flux.
        // Stats disjonction (diagnostic) : les indices label-keyed sont RÉUTILISÉS
        // entre couches → un buffer DÉDIÉ par couche/flux quand la collecte est
        // armée, même usage dans topk/gathers (valeurs identiques).
        let stats_active = expert_indices_collection_active();
        let mut collected: [Option<Buffer>; 2] = [None, None];
        for (stream, slot) in slots.into_iter().enumerate() {
            let _namespace = install_scratch_namespace(slot);
            let scratch = self.allocate_moe_shared_scratch(
                top_k,
                expert_count,
                inter_dim,
                out_dim,
                shared_inter_dim,
            )?;
            let indices_buffer = if stats_active {
                let dedicated = self.uncached_u32_buffer(top_k, "lightbatch_expert_indices")?;
                collected[stream] = Some(dedicated.clone());
                dedicated
            } else {
                scratch.indices.clone()
            };
            let in_row = self.private_f32_buffer(in_dim, "lightbatch_moe_in")?;
            let res_row = self.private_f32_buffer(out_dim, "lightbatch_moe_res")?;
            let out_row = self.private_f32_buffer(out_dim, "lightbatch_moe_out")?;
            let sdown_row = self.private_f32_buffer(out_dim, "lightbatch_moe_sdown_row")?;
            let in_offset = byte_offset_f32(
                stream
                    .checked_mul(in_dim)
                    .ok_or_else(|| InferError::Dimension("MoE duo row déborde".to_string()))?,
                "MoE duo in offset",
            )?;
            let router_offset = byte_offset_f32(
                stream.checked_mul(expert_count).ok_or_else(|| {
                    InferError::Dimension("MoE duo router row déborde".to_string())
                })?,
                "MoE duo router offset",
            )?;
            self.encode_copy_with_offsets(
                encoder,
                &router2,
                router_offset,
                &scratch.router,
                0,
                expert_count,
            )?;
            self.encode_copy_with_offsets(encoder, input2, in_offset, &in_row, 0, in_dim)?;
            self.encode_copy_with_offsets(encoder, residual2, in_offset, &res_row, 0, out_dim)?;
            self.encode_copy_with_offsets(encoder, &sdown2, in_offset, &sdown_row, 0, out_dim)?;
            self.encode_topk_softmax(
                encoder,
                owned_buffers,
                &scratch.router,
                &indices_buffer,
                &scratch.scores,
                expert_count,
                top_k,
            )?;
            let shared_gate_out = self.encode_matmul_weight_buffers(
                encoder,
                &in_row,
                1,
                in_dim,
                &weights.shared_gate,
                &scratch.shared_gate,
                false,
            )?;
            if shared_gate_out != 1 {
                return Err(InferError::Dimension(format!(
                    "MoE duo shared gate scalaire sort {shared_gate_out}, attendu 1"
                )));
            }
            if !self.encode_gather_gate_up_swiglu(
                encoder,
                owned_buffers,
                &in_row,
                1,
                &weights.stacked.gate,
                &weights.stacked.up,
                &indices_buffer,
                top_k,
                &scratch.hidden,
            )? {
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    &in_row,
                    1,
                    &weights.stacked.gate,
                    &indices_buffer,
                    top_k,
                    &scratch.gate,
                )?;
                self.encode_gather_matmul(
                    encoder,
                    owned_buffers,
                    &in_row,
                    1,
                    &weights.stacked.up,
                    &indices_buffer,
                    top_k,
                    &scratch.up,
                )?;
                self.encode_swiglu(
                    encoder,
                    owned_buffers,
                    &scratch.gate,
                    &scratch.up,
                    &scratch.hidden,
                    checked_len(top_k, inter_dim, "MoE duo swiglu")?,
                )?;
            }
            self.encode_gather_matmul(
                encoder,
                owned_buffers,
                &scratch.hidden,
                top_k,
                &weights.stacked.down,
                &indices_buffer,
                top_k,
                &scratch.down,
            )?;
            self.encode_weighted_sum_add_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                &res_row,
                &out_row,
                top_k,
                out_dim,
            )?;
            self.encode_add_sigmoid_scaled(
                encoder,
                &sdown_row,
                &scratch.shared_gate,
                &out_row,
                out_dim,
            )?;
            self.encode_copy_with_offsets(encoder, &out_row, 0, output2, in_offset, out_dim)?;
        }
        if let [Some(first), Some(second)] = collected {
            push_expert_indices_pair([first, second], top_k);
        }
        Ok(())
    }
}
