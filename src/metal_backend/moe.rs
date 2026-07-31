//! Orchestration Metal du MoE routé.

use super::*;

#[derive(Clone, Copy, Debug)]
enum MoeProjection {
    Gate,
    Up,
    Down,
}

impl MoeProjection {
    fn affine_weight(self, expert: &GatedMlp) -> Result<&AffineQuantizedTensor> {
        let (gate, up, down) = expert.projections();
        let linear = match self {
            Self::Gate => gate,
            Self::Up => up,
            Self::Down => down,
        };
        ensure_biasless(linear, self.name())?;
        match linear.weight() {
            LinearWeight::AffineQuantized(weight) => Ok(weight),
            LinearWeight::Dense(_) => Err(InferError::Config(format!(
                "expert MoE {} non affine quantifié",
                self.name()
            ))),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        }
    }

    fn packed_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_packed",
            Self::Up => "moe_up_packed",
            Self::Down => "moe_down_packed",
        }
    }

    fn scales_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_scales",
            Self::Up => "moe_up_scales",
            Self::Down => "moe_down_scales",
        }
    }

    fn biases_label(self) -> &'static str {
        match self {
            Self::Gate => "moe_gate_biases",
            Self::Up => "moe_up_biases",
            Self::Down => "moe_down_biases",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct MoeStackTraceContext {
    phase: &'static str,
    layer_index: Option<usize>,
}

impl MoeStackTraceContext {
    pub(super) const fn runtime() -> Self {
        Self {
            phase: "runtime",
            layer_index: None,
        }
    }

    pub(super) const fn cpu_release(layer_index: Option<usize>) -> Self {
        Self {
            phase: "cpu_release",
            layer_index,
        }
    }

    pub(super) const fn prefill(layer_index: usize) -> Self {
        Self {
            phase: "prefill",
            layer_index: Some(layer_index),
        }
    }
}

impl MetalExecutor {
    /// Exécute les experts MoE sélectionnés en une seule commande Metal.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les experts ont des biais, des formes incompatibles,
    /// ou si une commande Metal échoue.
    pub(crate) fn moe_gated_topk(
        &self,
        input: &Tensor,
        experts: &[GatedMlp],
        weighted_top: &[(usize, f32)],
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "MoE Metal attend batch=1, reçu {batch}"
            )));
        }
        if weighted_top.is_empty() {
            return Err(InferError::Config("MoE Metal top-k vide".to_string()));
        }
        if let Ok(stacked) = self.stacked_moe_buffers(experts) {
            let activation = experts
                .first()
                .map_or(crate::Activation::Silu, GatedMlp::activation);
            return self.moe_gated_topk_stacked(input, weighted_top, &stacked, activation);
        }
        let Some((first_expert, _)) = weighted_top.first() else {
            return Err(InferError::Config("MoE Metal top-k vide".to_string()));
        };
        let first = experts.get(*first_expert).ok_or_else(|| {
            InferError::Dimension(format!("expert MoE {first_expert} hors bornes"))
        })?;
        let (gate, up, down) = first.projections();
        let inter_dim = linear_out_dim(gate.weight())?;
        let up_dim = linear_out_dim(up.weight())?;
        let out_dim = linear_out_dim(down.weight())?;
        let down_in_dim = linear_in_dim(down.weight())?;
        if inter_dim != up_dim || inter_dim != down_in_dim {
            return Err(InferError::Dimension(format!(
                "formes MoE Metal incompatibles: gate={inter_dim}, up={up_dim}, down_in={down_in_dim}"
            )));
        }

        let input_buffer = self.upload_f32_buffer(input.data(), "moe_input")?;
        let gate_buffer = self.private_f32_buffer(inter_dim, "moe_gate")?;
        let up_buffer = self.private_f32_buffer(inter_dim, "moe_up")?;
        let hidden_buffer = self.private_f32_buffer(inter_dim, "moe_hidden")?;
        let down_buffer = self.private_f32_buffer(out_dim, "moe_down")?;
        let zeros = vec![0.0_f32; out_dim];
        let output_buffer = self.upload_f32_buffer(&zeros, "moe_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        for (expert_idx, scale) in weighted_top {
            let expert = experts.get(*expert_idx).ok_or_else(|| {
                InferError::Dimension(format!("expert MoE {expert_idx} hors bornes"))
            })?;
            let (gate, up, down) = expert.projections();
            ensure_biasless(gate, "gate")?;
            ensure_biasless(up, "up")?;
            ensure_biasless(down, "down")?;
            let gate_dim = self.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                1,
                in_dim,
                gate.weight(),
                &gate_buffer,
            )?;
            let up_dim = self.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                1,
                in_dim,
                up.weight(),
                &up_buffer,
            )?;
            if gate_dim != inter_dim || up_dim != inter_dim {
                return Err(InferError::Dimension(format!(
                    "expert MoE {expert_idx} inter_dim incohérent: gate={gate_dim}, up={up_dim}, attendu={inter_dim}"
                )));
            }
            match expert.activation() {
                crate::Activation::Silu => self.encode_swiglu(
                    encoder,
                    &mut owned_buffers,
                    &gate_buffer,
                    &up_buffer,
                    &hidden_buffer,
                    inter_dim,
                )?,
                crate::Activation::GeluTanh => self.encode_geglu_tanh(
                    encoder,
                    &mut owned_buffers,
                    &gate_buffer,
                    &up_buffer,
                    &hidden_buffer,
                    inter_dim,
                )?,
            }
            let down_dim = self.encode_matmul_weight(
                encoder,
                &mut owned_buffers,
                &hidden_buffer,
                1,
                inter_dim,
                down.weight(),
                &down_buffer,
            )?;
            if down_dim != out_dim {
                return Err(InferError::Dimension(format!(
                    "expert MoE {expert_idx} out_dim incohérent: {down_dim}, attendu={out_dim}"
                )));
            }
            self.encode_accumulate_scaled(
                encoder,
                &mut owned_buffers,
                &down_buffer,
                &output_buffer,
                *scale,
                out_dim,
            )?;
        }
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    /// Route et exécute un MoE top-k sans rapatrier les logits du routeur.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le routeur ou les experts ne sont pas biasless et
    /// quantifiés comme attendu par le chemin empilé Metal.
    pub(crate) fn moe_gated_router_topk(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
    ) -> Result<Tensor> {
        ensure_biasless(router, "router")?;
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "MoE Metal router attend batch=1, reçu {batch}"
            )));
        }
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "routeur MoE experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if in_dim != stacked.gate.in_dim || in_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE router input=[{batch},{in_dim}], gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE router inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        let inter_dim = stacked.gate.out_dim;
        let out_dim = stacked.down.out_dim;

        let input_buffer = self.upload_f32_buffer(input.data(), "moe_router_input")?;
        let router_buffer = self.private_f32_buffer(expert_count, "moe_router_logits")?;
        let indices_buffer = self.private_u32_buffer(top_k, "moe_router_indices")?;
        let scores_buffer = self.private_f32_buffer(top_k, "moe_router_scores")?;
        let gate_buffer = self.private_f32_buffer(
            checked_len(top_k, inter_dim, "moe router gate")?,
            "moe_router_gate",
        )?;
        let up_buffer = self.private_f32_buffer(
            checked_len(top_k, inter_dim, "moe router up")?,
            "moe_router_up",
        )?;
        let hidden_buffer = self.private_f32_buffer(
            checked_len(top_k, inter_dim, "moe router hidden")?,
            "moe_router_hidden",
        )?;
        let down_buffer = self.private_f32_buffer(
            checked_len(top_k, out_dim, "moe router down")?,
            "moe_router_down",
        )?;
        let output_buffer = self.new_f32_buffer(out_dim, "moe_router_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let router_out_dim = self.encode_matmul_weight(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            batch,
            in_dim,
            router.weight(),
            &router_buffer,
        )?;
        if router_out_dim != expert_count {
            return Err(InferError::Dimension(format!(
                "routeur MoE Metal sort {router_out_dim}, attendu {expert_count}"
            )));
        }
        self.encode_topk_softmax(
            encoder,
            &mut owned_buffers,
            &router_buffer,
            &indices_buffer,
            &scores_buffer,
            expert_count,
            top_k,
        )?;
        if !self.encode_gather_gate_up_swiglu(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            1,
            &stacked.gate,
            &stacked.up,
            &indices_buffer,
            top_k,
            &hidden_buffer,
        )? {
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                1,
                &stacked.gate,
                &indices_buffer,
                top_k,
                &gate_buffer,
            )?;
            self.encode_gather_matmul(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                1,
                &stacked.up,
                &indices_buffer,
                top_k,
                &up_buffer,
            )?;
            self.encode_swiglu(
                encoder,
                &mut owned_buffers,
                &gate_buffer,
                &up_buffer,
                &hidden_buffer,
                checked_len(top_k, inter_dim, "moe router swiglu")?,
            )?;
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            top_k,
            &stacked.down,
            &indices_buffer,
            top_k,
            &down_buffer,
        )?;
        self.encode_weighted_sum_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &output_buffer,
            top_k,
            out_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    /// Route et exécute un MoE top-k avec shared expert sans rapatrier le gate.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les poids shared ou MoE ne correspondent pas aux
    /// formes attendues par le chemin Metal.
    pub(crate) fn moe_gated_router_topk_shared(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "MoE shared Metal router attend batch=1, reçu {batch}"
            )));
        }
        let out_dim = self.stacked_moe_buffers(experts)?.down.out_dim;
        let input_buffer = self.upload_f32_buffer(input.data(), "moe_shared_input")?;
        let output_buffer = self.new_f32_buffer(out_dim, "moe_shared_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_moe_shared(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            None,
            &output_buffer,
            in_dim,
            router,
            experts,
            top_k,
            shared_expert,
            shared_gate,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    /// Route et exécute deux lignes MoE shared via le chemin duo qmm2.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les poids ne sont pas éligibles au duo ou si Metal échoue.
    pub(crate) fn moe_gated_router_topk_shared_batch2(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<Tensor> {
        if !crate::runtime_flags::lightbatch_moe2_enabled() {
            return Err(InferError::Metal(
                "MoE shared batch2 désactivé par RETI_RUST_LIGHTBATCH_MOE2".to_string(),
            ));
        }
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 2 {
            return Err(InferError::Dimension(format!(
                "MoE shared batch2 attend batch=2, reçu {batch}"
            )));
        }
        let weights =
            self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;
        if !self.moe_shared_duo_eligible(&weights) {
            return Err(InferError::Dimension(
                "MoE shared batch2 poids non éligibles qmm2".to_string(),
            ));
        }
        let out_dim = weights.stacked.down.out_dim;
        let input_buffer = self.upload_f32_buffer(input.data(), "moe_shared_batch2_input")?;
        let output_len = checked_len(2, out_dim, "moe batch2 output")?;
        let residual_zeros = vec![0.0_f32; output_len];
        let residual_buffer = self.upload_f32_buffer(&residual_zeros, "moe_shared_batch2_zero")?;
        let output_buffer = self.new_f32_buffer(output_len, "moe_shared_batch2_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        self.encode_moe_shared_duo_buffers(
            encoder,
            &mut owned_buffers,
            &input_buffer,
            &residual_buffer,
            &output_buffer,
            in_dim,
            &weights,
            top_k,
            [0, 1],
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![2, out_dim], output)
    }

    /// Route et exécute `batch` lignes MoE shared via les kernels rows génériques.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le batch est vide, si les poids sont incompatibles,
    /// ou si Metal échoue.
    pub(crate) fn moe_gated_router_topk_shared_rows(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch == 0 {
            return Err(InferError::Dimension(
                "MoE shared rows attend un batch non vide".to_string(),
            ));
        }
        let weights =
            self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;
        let out_dim = weights.stacked.down.out_dim;
        let input_buffer = self.upload_f32_buffer(input.data(), "moe_shared_rows_input")?;
        let output_len = checked_len(batch, out_dim, "moe shared rows output")?;
        let output_buffer = self.new_f32_buffer(output_len, "moe_shared_rows_output")?;
        let mut owned_buffers = Vec::new();

        // Brick #13 : route via le coop tensor-core (gather_qmm porté) quand activé —
        // le MoE des couches linéaires l'ignorait (encode_moe_shared_buffers_rows non-coop),
        // alors que le chemin full-attn l'utilise. C'était 66% du prefill (post_linear).
        if moe_coop_enabled() && weights.coop_compatible() {
            self.moe_shared_rows_coop(
                &input_buffer,
                None,
                &output_buffer,
                batch,
                in_dim,
                &weights,
                top_k,
            )?;
        } else {
            let command_buffer = self.queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();
            let encoder_guard = EncoderEndGuard::new(encoder);
            self.encode_moe_shared_buffers_rows(
                encoder,
                &mut owned_buffers,
                &input_buffer,
                None,
                &output_buffer,
                batch,
                in_dim,
                &weights,
                top_k,
            )?;
            encoder_guard.end();
            commit_and_wait(command_buffer)?;
        }

        let output = read_f32_buffer(&output_buffer, output_len)?;
        Tensor::from_vec(vec![batch, out_dim], output)
    }

    pub(super) fn moe_gated_topk_stacked(
        &self,
        input: &Tensor,
        weighted_top: &[(usize, f32)],
        stacked: &StackedMoeBuffers,
        activation: crate::Activation,
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 || in_dim != stacked.gate.in_dim || in_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE stacked input=[{batch},{in_dim}], gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE stacked inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        let topk = weighted_top.len();
        let inter_dim = stacked.gate.out_dim;
        let out_dim = stacked.down.out_dim;
        let mut indices = Vec::with_capacity(topk);
        let mut scores = Vec::with_capacity(topk);
        for (expert_idx, score) in weighted_top {
            if *expert_idx >= stacked.gate.experts {
                return Err(InferError::Dimension(format!(
                    "expert MoE {expert_idx} hors bornes pour {} experts",
                    stacked.gate.experts
                )));
            }
            indices.push(checked_u32(*expert_idx, "expert_idx")?);
            scores.push(*score);
        }

        let input_buffer = self.upload_f32_buffer(input.data(), "moe_stack_input")?;
        let indices_buffer = self.upload_u32_buffer(&indices, "moe_stack_indices")?;
        let scores_buffer = self.upload_f32_buffer(&scores, "moe_stack_scores")?;
        let gate_buffer = self.private_f32_buffer(
            checked_len(topk, inter_dim, "moe stack gate")?,
            "moe_stack_gate",
        )?;
        let up_buffer = self.private_f32_buffer(
            checked_len(topk, inter_dim, "moe stack up")?,
            "moe_stack_up",
        )?;
        let hidden_buffer = self.private_f32_buffer(
            checked_len(topk, inter_dim, "moe stack hidden")?,
            "moe_stack_hidden",
        )?;
        let down_buffer = self.private_f32_buffer(
            checked_len(topk, out_dim, "moe stack down")?,
            "moe_stack_down",
        )?;
        let output_buffer = self.new_f32_buffer(out_dim, "moe_stack_output")?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        match activation {
            crate::Activation::Silu => {
                if !self.encode_gather_gate_up_swiglu(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    1,
                    &stacked.gate,
                    &stacked.up,
                    &indices_buffer,
                    topk,
                    &hidden_buffer,
                )? {
                    self.encode_gather_matmul(
                        encoder,
                        &mut owned_buffers,
                        &input_buffer,
                        1,
                        &stacked.gate,
                        &indices_buffer,
                        topk,
                        &gate_buffer,
                    )?;
                    self.encode_gather_matmul(
                        encoder,
                        &mut owned_buffers,
                        &input_buffer,
                        1,
                        &stacked.up,
                        &indices_buffer,
                        topk,
                        &up_buffer,
                    )?;
                    self.encode_swiglu(
                        encoder,
                        &mut owned_buffers,
                        &gate_buffer,
                        &up_buffer,
                        &hidden_buffer,
                        checked_len(topk, inter_dim, "moe stack swiglu")?,
                    )?;
                }
            }
            crate::Activation::GeluTanh => {
                self.encode_gather_matmul(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    1,
                    &stacked.gate,
                    &indices_buffer,
                    topk,
                    &gate_buffer,
                )?;
                self.encode_gather_matmul(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    1,
                    &stacked.up,
                    &indices_buffer,
                    topk,
                    &up_buffer,
                )?;
                self.encode_geglu_tanh(
                    encoder,
                    &mut owned_buffers,
                    &gate_buffer,
                    &up_buffer,
                    &hidden_buffer,
                    checked_len(topk, inter_dim, "moe stack geglu")?,
                )?;
            }
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            topk,
            &stacked.down,
            &indices_buffer,
            topk,
            &down_buffer,
        )?;
        self.encode_weighted_sum_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &output_buffer,
            topk,
            out_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, out_dim)?;
        Tensor::from_vec(vec![1, out_dim], output)
    }

    pub(crate) fn moe_gated_topk_batch(
        &self,
        input: &Tensor,
        experts: &[GatedMlp],
        weighted_top_rows: &[Vec<(usize, f32)>],
    ) -> Result<Tensor> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch == 0 {
            return Err(InferError::Dimension("MoE batch vide".to_string()));
        }
        if weighted_top_rows.len() != batch {
            return Err(InferError::Dimension(format!(
                "MoE batch rows={} mais top rows={}",
                batch,
                weighted_top_rows.len()
            )));
        }
        let topk = weighted_top_rows
            .first()
            .map(Vec::len)
            .ok_or_else(|| InferError::Config("MoE batch top-k vide".to_string()))?;
        if weighted_top_rows.iter().any(|row| row.len() != topk) {
            return Err(InferError::Dimension(
                "MoE batch top-k variable par ligne".to_string(),
            ));
        }

        let stacked = self.stacked_moe_buffers(experts)?;
        ensure_valid_top_k(topk, stacked.gate.experts)?;
        if in_dim != stacked.gate.in_dim || in_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE batch input=[{batch},{in_dim}], gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE batch inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }

        let total_topk = checked_len(batch, topk, "MoE batch topk total")?;
        let inter_dim = stacked.gate.out_dim;
        let out_dim = stacked.down.out_dim;
        let mut indices = Vec::with_capacity(total_topk);
        let mut scores = Vec::with_capacity(total_topk);
        for row in weighted_top_rows {
            for (expert_idx, score) in row {
                if *expert_idx >= stacked.gate.experts {
                    return Err(InferError::Dimension(format!(
                        "expert MoE {expert_idx} hors bornes pour {} experts",
                        stacked.gate.experts
                    )));
                }
                indices.push(checked_u32(*expert_idx, "expert_idx batch")?);
                scores.push(*score);
            }
        }

        let input_buffer = self.upload_f32_buffer(input.data(), "moe_batch_input")?;
        let indices_buffer = self.upload_u32_buffer(&indices, "moe_batch_indices")?;
        let scores_buffer = self.upload_f32_buffer(&scores, "moe_batch_scores")?;
        let gate_buffer = self.private_f32_buffer(
            checked_len(total_topk, inter_dim, "moe batch gate")?,
            "moe_batch_gate",
        )?;
        let up_buffer = self.private_f32_buffer(
            checked_len(total_topk, inter_dim, "moe batch up")?,
            "moe_batch_up",
        )?;
        let hidden_buffer = self.private_f32_buffer(
            checked_len(total_topk, inter_dim, "moe batch hidden")?,
            "moe_batch_hidden",
        )?;
        let down_buffer = self.private_f32_buffer(
            checked_len(total_topk, out_dim, "moe batch down")?,
            "moe_batch_down",
        )?;
        let output_buffer = self.new_f32_buffer(
            checked_len(batch, out_dim, "moe batch output")?,
            "moe_batch_output",
        )?;
        let mut owned_buffers = Vec::new();

        let command_buffer = self.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        // NOTE: l'activation vient des experts (GeGLU pour Gemma, SwiGLU sinon) ;
        // l'ignorer calculait le MoE Gemma 4 en SiLU (divergence prefill résident).
        let activation = experts
            .first()
            .map_or(crate::Activation::Silu, GatedMlp::activation);
        match activation {
            crate::Activation::Silu => {
                if !self.encode_gather_gate_up_swiglu(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    batch,
                    &stacked.gate,
                    &stacked.up,
                    &indices_buffer,
                    total_topk,
                    &hidden_buffer,
                )? {
                    self.encode_gather_matmul(
                        encoder,
                        &mut owned_buffers,
                        &input_buffer,
                        batch,
                        &stacked.gate,
                        &indices_buffer,
                        total_topk,
                        &gate_buffer,
                    )?;
                    self.encode_gather_matmul(
                        encoder,
                        &mut owned_buffers,
                        &input_buffer,
                        batch,
                        &stacked.up,
                        &indices_buffer,
                        total_topk,
                        &up_buffer,
                    )?;
                    self.encode_swiglu(
                        encoder,
                        &mut owned_buffers,
                        &gate_buffer,
                        &up_buffer,
                        &hidden_buffer,
                        checked_len(total_topk, inter_dim, "moe batch swiglu")?,
                    )?;
                }
            }
            crate::Activation::GeluTanh => {
                self.encode_gather_matmul(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    batch,
                    &stacked.gate,
                    &indices_buffer,
                    total_topk,
                    &gate_buffer,
                )?;
                self.encode_gather_matmul(
                    encoder,
                    &mut owned_buffers,
                    &input_buffer,
                    batch,
                    &stacked.up,
                    &indices_buffer,
                    total_topk,
                    &up_buffer,
                )?;
                self.encode_geglu_tanh(
                    encoder,
                    &mut owned_buffers,
                    &gate_buffer,
                    &up_buffer,
                    &hidden_buffer,
                    checked_len(total_topk, inter_dim, "moe batch geglu")?,
                )?;
            }
        }
        self.encode_gather_matmul(
            encoder,
            &mut owned_buffers,
            &hidden_buffer,
            total_topk,
            &stacked.down,
            &indices_buffer,
            total_topk,
            &down_buffer,
        )?;
        self.encode_weighted_sum_grouped_topk(
            encoder,
            &mut owned_buffers,
            &down_buffer,
            &scores_buffer,
            &output_buffer,
            batch,
            topk,
            out_dim,
        )?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;

        let output = read_f32_buffer(&output_buffer, batch * out_dim)?;
        Tensor::from_vec(vec![batch, out_dim], output)
    }

    pub(super) fn stacked_moe_buffers(&self, experts: &[GatedMlp]) -> Result<StackedMoeBuffers> {
        self.stacked_moe_buffers_with_context(experts, MoeStackTraceContext::runtime())
    }

    pub(super) fn stacked_moe_buffers_with_context(
        &self,
        experts: &[GatedMlp],
        trace_context: MoeStackTraceContext,
    ) -> Result<StackedMoeBuffers> {
        if experts.is_empty() {
            return Err(InferError::Config("MoE sans expert".to_string()));
        }
        let key = moe_weight_key(experts)?;
        {
            let stacks = self
                .moe_stacks
                .lock()
                .map_err(|_| InferError::Metal("cache MoE Metal empoisonné".to_string()))?;
            if let Some(buffers) = stacks.get(&key) {
                trace_moe_stack_resolution(trace_context, true, &key, buffers);
                return Ok(buffers.clone());
            }
        }
        let buffers = StackedMoeBuffers {
            gate: self.build_stacked_affine(experts, MoeProjection::Gate)?,
            up: self.build_stacked_affine(experts, MoeProjection::Up)?,
            down: self.build_stacked_affine(experts, MoeProjection::Down)?,
        };
        let mut stacks = self
            .moe_stacks
            .lock()
            .map_err(|_| InferError::Metal("cache MoE Metal empoisonné".to_string()))?;
        stacks.insert(key.clone(), buffers.clone());
        trace_moe_stack_resolution(trace_context, false, &key, &buffers);
        Ok(buffers)
    }

    fn build_stacked_affine(
        &self,
        experts: &[GatedMlp],
        projection: MoeProjection,
    ) -> Result<StackedAffineBuffers> {
        let first = projection.affine_weight(&experts[0])?;
        let [out_dim, in_dim] = first.shape() else {
            return Err(InferError::Dimension(format!(
                "poids expert attendu rang 2, reçu {:?}",
                first.shape()
            )));
        };
        let [packed_rows, packed_cols] = first.packed_shape() else {
            return Err(InferError::Dimension(format!(
                "poids expert packed attendu rang 2, reçu {:?}",
                first.packed_shape()
            )));
        };
        if *packed_rows != *out_dim {
            return Err(InferError::Dimension(format!(
                "poids expert packed_rows={packed_rows}, out_dim={out_dim}"
            )));
        }
        let groups = in_dim
            .checked_div(first.group_size())
            .ok_or_else(|| InferError::Metal("group_size quantifié nul".to_string()))?;
        let expected_affine_shape = [*out_dim, groups];
        let mut weights = Vec::with_capacity(experts.len());
        for (idx, expert) in experts.iter().enumerate() {
            let weight = projection.affine_weight(expert)?;
            if weight.shape() != [*out_dim, *in_dim]
                || weight.packed_shape() != [*packed_rows, *packed_cols]
                || weight.group_size() != first.group_size()
                || weight.bits() != first.bits()
                || weight.scales_shape() != expected_affine_shape
                || weight.biases_shape() != expected_affine_shape
            {
                return Err(InferError::Dimension(format!(
                    "expert MoE {idx} {:?} incompatible avec le premier expert",
                    projection
                )));
            }
            weights.push(weight);
        }
        let direct = contiguous_metal_expert_stacks(&weights)?;
        let (packed, packed_offset, scales, scales_offset, biases, biases_offset) = match direct {
            Some(direct) => direct,
            None => {
                let mut packed = Vec::with_capacity(experts.len() * first.packed_data().len());
                let mut scales = Vec::with_capacity(experts.len() * first.scales_len());
                let mut biases = Vec::with_capacity(experts.len() * first.biases_len());
                for weight in &weights {
                    packed.extend_from_slice(weight.packed_data());
                    scales.extend_from_slice(weight.scales_f32().as_ref());
                    biases.extend_from_slice(weight.biases_f32().as_ref());
                }
                (
                    self.buffer_from_u32(&packed, projection.packed_label())?,
                    0,
                    self.buffer_from_f32_as_bf16(&scales, projection.scales_label())?,
                    0,
                    self.buffer_from_f32_as_bf16(&biases, projection.biases_label())?,
                    0,
                )
            }
        };
        Ok(StackedAffineBuffers {
            packed,
            packed_offset,
            scales,
            scales_offset,
            biases,
            biases_offset,
            experts: experts.len(),
            out_dim: *out_dim,
            in_dim: *in_dim,
            packed_cols: *packed_cols,
            group_size: first.group_size(),
            bits: first.bits(),
            groups,
        })
    }
}

type DirectExpertStacks = (
    metal::Buffer,
    NSUInteger,
    metal::Buffer,
    NSUInteger,
    metal::Buffer,
    NSUInteger,
);

fn contiguous_metal_expert_stacks(
    weights: &[&AffineQuantizedTensor],
) -> Result<Option<DirectExpertStacks>> {
    let Some(first) = weights.first() else {
        return Ok(None);
    };
    let Some((first_buffer, first_offset)) = first.packed_metal_view() else {
        return Ok(None);
    };
    let Some((first_scales_buffer, first_scales_offset)) = first.scales_metal_view() else {
        return Ok(None);
    };
    let Some((first_biases_buffer, first_biases_offset)) = first.biases_metal_view() else {
        return Ok(None);
    };
    let packed_stride_bytes = first
        .packed_data()
        .len()
        .checked_mul(std::mem::size_of::<u32>())
        .ok_or_else(|| InferError::Shape("stride stack MoE trop grand".to_string()))?;
    let scales_stride_bytes = first
        .scales_len()
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| InferError::Shape("stride scales MoE trop grand".to_string()))?;
    let biases_stride_bytes = first
        .biases_len()
        .checked_mul(std::mem::size_of::<u16>())
        .ok_or_else(|| InferError::Shape("stride biases MoE trop grand".to_string()))?;
    for (index, weight) in weights.iter().enumerate() {
        let Some((buffer, offset)) = weight.packed_metal_view() else {
            return Ok(None);
        };
        let Some((scales_buffer, scales_offset)) = weight.scales_metal_view() else {
            return Ok(None);
        };
        let Some((biases_buffer, biases_offset)) = weight.biases_metal_view() else {
            return Ok(None);
        };
        let expected_packed_offset = index
            .checked_mul(packed_stride_bytes)
            .and_then(|value| first_offset.checked_add(value))
            .ok_or_else(|| InferError::Shape("offset stack MoE trop grand".to_string()))?;
        let expected_scales_offset = index
            .checked_mul(scales_stride_bytes)
            .and_then(|value| first_scales_offset.checked_add(value))
            .ok_or_else(|| InferError::Shape("offset scales MoE trop grand".to_string()))?;
        let expected_biases_offset = index
            .checked_mul(biases_stride_bytes)
            .and_then(|value| first_biases_offset.checked_add(value))
            .ok_or_else(|| InferError::Shape("offset biases MoE trop grand".to_string()))?;
        if buffer.contents() != first_buffer.contents()
            || offset != expected_packed_offset
            || scales_buffer.contents() != first_scales_buffer.contents()
            || scales_offset != expected_scales_offset
            || biases_buffer.contents() != first_biases_buffer.contents()
            || biases_offset != expected_biases_offset
        {
            return Ok(None);
        }
    }
    Ok(Some((
        first_buffer.clone(),
        checked_nsuint(first_offset, "offset stack MoE")?,
        first_scales_buffer.clone(),
        checked_nsuint(first_scales_offset, "offset scales stack MoE")?,
        first_biases_buffer.clone(),
        checked_nsuint(first_biases_offset, "offset biases stack MoE")?,
    )))
}

fn moe_weight_key(experts: &[GatedMlp]) -> Result<MoeWeightKey> {
    let mut weights = Vec::with_capacity(experts.len());
    for expert in experts {
        weights.push([
            MoeProjection::Gate.affine_weight(expert)?.weight_id(),
            MoeProjection::Up.affine_weight(expert)?.weight_id(),
            MoeProjection::Down.affine_weight(expert)?.weight_id(),
        ]);
    }
    Ok(MoeWeightKey { weights })
}

fn trace_moe_stack_resolution(
    context: MoeStackTraceContext,
    cache_hit: bool,
    key: &MoeWeightKey,
    buffers: &StackedMoeBuffers,
) {
    if !crate::runtime_flags::moe_drop_trace_enabled() {
        return;
    }
    let layer = context
        .layer_index
        .map_or_else(|| "na".to_string(), |index| index.to_string());
    let cache = if cache_hit { "hit" } else { "miss" };
    eprintln!(
        "moe_drop_trace phase={} layer={layer} cache={cache} expert_ids=0..{} \
         weight_ids={:?} \
         gate_packed_ptr={:p} gate_packed_bytes={} \
         gate_scales_ptr={:p} gate_scales_bytes={} \
         gate_biases_ptr={:p} gate_biases_bytes={} \
         up_packed_ptr={:p} up_packed_bytes={} \
         up_scales_ptr={:p} up_scales_bytes={} \
         up_biases_ptr={:p} up_biases_bytes={} \
         down_packed_ptr={:p} down_packed_bytes={} \
         down_scales_ptr={:p} down_scales_bytes={} \
         down_biases_ptr={:p} down_biases_bytes={}",
        context.phase,
        key.weights.len(),
        key.weights,
        buffers.gate.packed.as_ptr(),
        buffers.gate.packed.length(),
        buffers.gate.scales.as_ptr(),
        buffers.gate.scales.length(),
        buffers.gate.biases.as_ptr(),
        buffers.gate.biases.length(),
        buffers.up.packed.as_ptr(),
        buffers.up.packed.length(),
        buffers.up.scales.as_ptr(),
        buffers.up.scales.length(),
        buffers.up.biases.as_ptr(),
        buffers.up.biases.length(),
        buffers.down.packed.as_ptr(),
        buffers.down.packed.length(),
        buffers.down.scales.as_ptr(),
        buffers.down.scales.length(),
        buffers.down.biases.as_ptr(),
        buffers.down.biases.length(),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_affine(marker: u32) -> Result<AffineQuantizedTensor> {
        AffineQuantizedTensor::new(
            &[1, 1],
            vec![marker],
            Tensor::from_vec(vec![1, 1], vec![0.5])?,
            Tensor::from_vec(vec![1, 1], vec![0.0])?,
            8,
            4,
        )
    }

    fn test_expert(marker: u32) -> Result<GatedMlp> {
        let linear = |offset| {
            Linear::from_weight(
                LinearWeight::AffineQuantized(test_affine(marker + offset)?),
                None,
            )
        };
        Ok(GatedMlp::new(linear(0)?, linear(1)?, linear(2)?))
    }

    #[test]
    fn moe_weight_key_survives_cpu_drop_and_preserves_expert_order() -> Result<()> {
        let mut experts = vec![test_expert(10)?, test_expert(20)?];
        let key_before_drop = moe_weight_key(&experts)?;
        let mut reversed = experts.clone();
        reversed.reverse();
        let reversed_key = moe_weight_key(&reversed)?;
        assert_ne!(key_before_drop, reversed_key);

        let mut released = Vec::new();
        for expert in &mut experts {
            expert.release_affine_cpu_data(&mut released);
        }

        assert_eq!(key_before_drop, moe_weight_key(&experts)?);
        assert_eq!(released.len(), 6);
        Ok(())
    }
}
