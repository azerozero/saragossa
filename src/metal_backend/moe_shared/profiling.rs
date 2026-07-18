//! Profilage des segments du MoE shared.

use super::*;

impl MetalExecutor {
    /// Microbenche les segments du MoE shared (route, gate/up, down, tails).
    ///
    /// Mesure chaque segment isolé (warmup 8, 64 itérations, un command
    /// buffer par itération) sur une route figée par le premier top-k ;
    /// `overhead_ms` (coût commit+wait à vide) est soustrait pour la
    /// colonne « pur » du rapport.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `input` n'est pas batch=1, si une dimension est
    /// incompatible ou si un encodage échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "microbench MoE: dimensions, poids et overhead restent explicites"
    )]
    pub(crate) fn profile_moe_shared_segments(
        &self,
        input: &Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &GatedMlp,
        shared_gate: &Linear,
        overhead_ms: f64,
    ) -> Result<String> {
        let (batch, in_dim) = input.as_matrix()?;
        if batch != 1 {
            return Err(InferError::Dimension(format!(
                "MoE split attend batch=1, reçu {batch}"
            )));
        }
        let weights =
            self.resolve_moe_shared_weights(router, experts, shared_expert, shared_gate)?;
        let shape = self.check_moe_shared_buffer_shapes(in_dim, &weights, top_k)?;
        let scratch = self.allocate_moe_shared_scratch(
            top_k,
            shape.expert_count,
            shape.inter_dim,
            shape.out_dim,
            shape.shared_inter_dim,
        )?;
        let input_buffer = self.upload_f32_buffer(input.data(), "moe_split_input")?;
        let output_buffer = self.new_f32_buffer(shape.out_dim, "moe_split_output")?;
        let residual_zeros = vec![0.0_f32; shape.out_dim];
        let residual_buffer = self.upload_f32_buffer(&residual_zeros, "moe_split_residual")?;
        let iters = 64_u32;
        let warmup = 8_u32;

        let route_topk = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            let router_out_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.router,
                &scratch.router,
                false,
            )?;
            if router_out_dim != shape.expert_count {
                return Err(InferError::Dimension(format!(
                    "split routeur sort {router_out_dim}, attendu {}",
                    shape.expert_count
                )));
            }
            self.encode_topk_softmax(
                encoder,
                owned,
                &scratch.router,
                &scratch.indices,
                &scratch.scores,
                shape.expert_count,
                top_k,
            )
        })?;

        let routed_gate_up = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            if self.encode_gather_gate_up_swiglu(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.gate,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.hidden,
            )? {
                return Ok(());
            }
            self.encode_gather_matmul(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.gate,
                &scratch.indices,
                top_k,
                &scratch.gate,
            )?;
            self.encode_gather_matmul(
                encoder,
                owned,
                &input_buffer,
                1,
                &weights.stacked.up,
                &scratch.indices,
                top_k,
                &scratch.up,
            )?;
            self.encode_swiglu(
                encoder,
                owned,
                &scratch.gate,
                &scratch.up,
                &scratch.hidden,
                checked_len(top_k, shape.inter_dim, "split routed swiglu")?,
            )
        })?;

        let shared_gate_ms = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            let projected_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_gate,
                &scratch.shared_gate,
                false,
            )?;
            if projected_gate_dim != 1 {
                return Err(InferError::Dimension(format!(
                    "split shared gate sort {projected_gate_dim}, attendu 1"
                )));
            }
            Ok(())
        })?;

        let shared_gate_up = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            if can_fuse_shared_gate_up_buffers(&weights.shared_gate_proj, &weights.shared_up_proj)
                && self.encode_gate_up_swiglu_fast_buffers(
                    encoder,
                    &input_buffer,
                    &weights.shared_gate_proj,
                    &weights.shared_up_proj,
                    &scratch.shared_hidden,
                    in_dim,
                )?
            {
                return Ok(());
            }
            let projected_shared_gate_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_gate_proj,
                &scratch.shared_proj_gate,
                false,
            )?;
            let projected_shared_up_dim = self.encode_matmul_weight_buffers(
                encoder,
                &input_buffer,
                1,
                in_dim,
                &weights.shared_up_proj,
                &scratch.shared_up,
                false,
            )?;
            if projected_shared_gate_dim != shape.shared_inter_dim
                || projected_shared_up_dim != shape.shared_inter_dim
            {
                return Err(InferError::Dimension(format!(
                    "split shared expert gate={projected_shared_gate_dim}, up={projected_shared_up_dim}, attendu {}",
                    shape.shared_inter_dim
                )));
            }
            self.encode_swiglu(
                encoder,
                owned,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                shape.shared_inter_dim,
            )
        })?;

        let routed_down = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            self.encode_gather_matmul(
                encoder,
                owned,
                &scratch.hidden,
                top_k,
                &weights.stacked.down,
                &scratch.indices,
                top_k,
                &scratch.down,
            )
        })?;

        let shared_down = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            let projected_shared_down_dim = self.encode_matmul_weight_buffers(
                encoder,
                &scratch.shared_hidden,
                1,
                shape.shared_inter_dim,
                &weights.shared_down_proj,
                &scratch.shared_down,
                false,
            )?;
            if projected_shared_down_dim != shape.out_dim {
                return Err(InferError::Dimension(format!(
                    "split shared down sort {projected_shared_down_dim}, attendu {}",
                    shape.out_dim
                )));
            }
            Ok(())
        })?;

        let tail_plain = profile_moe_segment(self, warmup, iters, |encoder, owned| {
            self.encode_weighted_sum_topk(
                encoder,
                owned,
                &scratch.down,
                &scratch.scores,
                &output_buffer,
                top_k,
                shape.out_dim,
            )?;
            self.encode_add_sigmoid_scaled(
                encoder,
                &scratch.shared_down,
                &scratch.shared_gate,
                &output_buffer,
                shape.out_dim,
            )
        })?;

        let tail_residual = profile_moe_segment(self, warmup, iters, |encoder, _owned| {
            self.encode_weighted_sum_add_shared_topk(
                encoder,
                &scratch.down,
                &scratch.scores,
                &residual_buffer,
                &scratch.shared_down,
                &scratch.shared_gate,
                &output_buffer,
                top_k,
                shape.out_dim,
            )
        })?;

        let pure = |segment_ms: f64| (segment_ms - overhead_ms).max(0.0);
        Ok(format!(
            "split MoE fixed-route ({iters} itér, ms+CB/pur): route {route_topk:.3}/{route_pure:.3}, \
             routed_gate_up {routed_gate_up:.3}/{routed_gate_up_pure:.3}, \
             shared_gate {shared_gate_ms:.3}/{shared_gate_pure:.3}, \
             shared_gate_up {shared_gate_up:.3}/{shared_gate_up_pure:.3}, \
             routed_down {routed_down:.3}/{routed_down_pure:.3}, \
             shared_down {shared_down:.3}/{shared_down_pure:.3}, \
             tail_plain {tail_plain:.3}/{tail_plain_pure:.3}, \
             tail_residual {tail_residual:.3}/{tail_residual_pure:.3}",
            route_pure = pure(route_topk),
            routed_gate_up_pure = pure(routed_gate_up),
            shared_gate_pure = pure(shared_gate_ms),
            shared_gate_up_pure = pure(shared_gate_up),
            routed_down_pure = pure(routed_down),
            shared_down_pure = pure(shared_down),
            tail_plain_pure = pure(tail_plain),
            tail_residual_pure = pure(tail_residual),
        ))
    }
}

/// Mesure la durée moyenne (ms) d'un segment encodé, commit+wait inclus.
///
/// # Errors
///
/// Renvoie une erreur si `iters == 0` ou si l'encodage/le commit échoue.
fn profile_moe_segment<F>(
    metal: &MetalExecutor,
    warmup: u32,
    iters: u32,
    mut encode: F,
) -> Result<f64>
where
    F: FnMut(&ComputeCommandEncoderRef, &mut Vec<Buffer>) -> Result<()>,
{
    if iters == 0 {
        return Err(InferError::Dimension("MoE split iters nul".to_string()));
    }
    for _ in 0..warmup {
        let command_buffer = metal.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        encode(encoder, &mut owned)?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
    }
    let started = std::time::Instant::now();
    for _ in 0..iters {
        let command_buffer = metal.queue.new_command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        let encoder_guard = EncoderEndGuard::new(encoder);
        let mut owned = Vec::new();
        encode(encoder, &mut owned)?;
        encoder_guard.end();
        commit_and_wait(command_buffer)?;
    }
    Ok(started.elapsed().as_secs_f64() * 1000.0 / f64::from(iters))
}
