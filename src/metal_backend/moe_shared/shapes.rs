//! Validation des formes des poids MoE shared et routed.

use super::*;

impl MetalExecutor {
    /// Valide les poids linéaires d'un MoE shared non résolu.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension diverge ou si un poids porte un biais.
    pub(super) fn check_moe_shared_linear_shapes<'a>(
        &self,
        in_dim: usize,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &'a GatedMlp,
        shared_gate: &Linear,
    ) -> Result<MoeSharedLinearShape<'a>> {
        ensure_biasless(router, "router")?;
        ensure_biasless(shared_gate, "shared_gate")?;
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if in_dim != stacked.gate.in_dim || in_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE shared in_dim={in_dim}, gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE shared inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        let (shared_gate_proj, shared_up_proj, shared_down_proj) = shared_expert.projections();
        ensure_biasless(shared_gate_proj, "shared_gate_proj")?;
        ensure_biasless(shared_up_proj, "shared_up_proj")?;
        ensure_biasless(shared_down_proj, "shared_down_proj")?;
        let shared_gate_out_dim = linear_out_dim(shared_gate.weight())?;
        if shared_gate_out_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate sort {shared_gate_out_dim}, attendu 1"
            )));
        }
        let shared_inter_dim = linear_out_dim(shared_gate_proj.weight())?;
        let shared_up_dim = linear_out_dim(shared_up_proj.weight())?;
        let shared_down_dim = linear_out_dim(shared_down_proj.weight())?;
        let shared_down_in_dim = linear_in_dim(shared_down_proj.weight())?;
        if shared_inter_dim != shared_up_dim || shared_inter_dim != shared_down_in_dim {
            return Err(InferError::Dimension(format!(
                "shared expert dims gate={shared_inter_dim}, up={shared_up_dim}, down_in={shared_down_in_dim}"
            )));
        }
        if shared_down_dim != stacked.down.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert out={shared_down_dim}, MoE out={}",
                stacked.down.out_dim
            )));
        }
        Ok(MoeSharedLinearShape {
            expert_count,
            inter_dim: stacked.gate.out_dim,
            out_dim: stacked.down.out_dim,
            shared_inter_dim,
            stacked,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
        })
    }

    /// Valide la cohérence dimensionnelle routeur/experts d'un MoE routed-only.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `top_k` est invalide ou si une dimension diverge.
    pub(in super::super) fn check_moe_routed_buffer_shapes(
        &self,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
    ) -> Result<MoeRoutedBufferShape> {
        let expert_count = self.linear_weight_out_dim(&weights.router);
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != weights.stacked.gate.experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed experts={expert_count}, poids experts={}",
                weights.stacked.gate.experts
            )));
        }
        if in_dim != weights.stacked.gate.in_dim || in_dim != weights.stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE routed in_dim={in_dim}, gate_in={}, up_in={}",
                weights.stacked.gate.in_dim, weights.stacked.up.in_dim
            )));
        }
        if weights.stacked.gate.out_dim != weights.stacked.up.out_dim
            || weights.stacked.down.in_dim != weights.stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE routed inter dims gate={} up={} down_in={}",
                weights.stacked.gate.out_dim,
                weights.stacked.up.out_dim,
                weights.stacked.down.in_dim
            )));
        }
        Ok(MoeRoutedBufferShape {
            expert_count,
            inter_dim: weights.stacked.gate.out_dim,
            out_dim: weights.stacked.down.out_dim,
        })
    }

    /// Valide routeur, experts empilés et shared expert ; renvoie les dimensions.
    ///
    /// Le gate shared doit sortir un scalaire (out_dim = 1) et le down du
    /// shared expert doit rejoindre `out_dim` des experts routés.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si `top_k` est invalide ou si une dimension diverge.
    pub(in super::super) fn check_moe_shared_buffer_shapes(
        &self,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<MoeSharedBufferShape> {
        let expert_count = self.linear_weight_out_dim(&weights.router);
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != weights.stacked.gate.experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE shared experts={expert_count}, poids experts={}",
                weights.stacked.gate.experts
            )));
        }
        if in_dim != weights.stacked.gate.in_dim || in_dim != weights.stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "MoE shared in_dim={in_dim}, gate_in={}, up_in={}",
                weights.stacked.gate.in_dim, weights.stacked.up.in_dim
            )));
        }
        if weights.stacked.gate.out_dim != weights.stacked.up.out_dim
            || weights.stacked.down.in_dim != weights.stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "MoE shared inter dims gate={} up={} down_in={}",
                weights.stacked.gate.out_dim,
                weights.stacked.up.out_dim,
                weights.stacked.down.in_dim
            )));
        }
        let shared_gate_out_dim = self.linear_weight_out_dim(&weights.shared_gate);
        if shared_gate_out_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate sort {shared_gate_out_dim}, attendu 1"
            )));
        }
        let shared_inter_dim = self.linear_weight_out_dim(&weights.shared_gate_proj);
        let shared_up_dim = self.linear_weight_out_dim(&weights.shared_up_proj);
        let shared_down_dim = self.linear_weight_out_dim(&weights.shared_down_proj);
        let shared_down_in_dim = self.linear_weight_in_dim(&weights.shared_down_proj);
        if shared_inter_dim != shared_up_dim || shared_inter_dim != shared_down_in_dim {
            return Err(InferError::Dimension(format!(
                "shared expert dims gate={shared_inter_dim}, up={shared_up_dim}, down_in={shared_down_in_dim}"
            )));
        }
        if shared_down_dim != weights.stacked.down.out_dim {
            return Err(InferError::Dimension(format!(
                "shared expert out={shared_down_dim}, MoE out={}",
                weights.stacked.down.out_dim
            )));
        }
        Ok(MoeSharedBufferShape {
            expert_count,
            inter_dim: weights.stacked.gate.out_dim,
            out_dim: weights.stacked.down.out_dim,
            shared_inter_dim,
        })
    }
}
