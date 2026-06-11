//! Gardes de formes des kernels full-attn Metal.

use super::*;

pub(super) struct TailMoeShape<'a> {
    pub(super) batch: usize,
    pub(super) hidden_dim: usize,
    pub(super) context_dim: usize,
    pub(super) norm_weight: &'a Tensor,
    pub(super) expert_count: usize,
    pub(super) inter_dim: usize,
    pub(super) stacked: StackedMoeBuffers,
}

pub(super) struct TailMoeSharedShape<'a> {
    pub(super) batch: usize,
    pub(super) hidden_dim: usize,
    pub(super) context_dim: usize,
    pub(super) norm_weight: &'a Tensor,
    pub(super) expert_count: usize,
    pub(super) inter_dim: usize,
    pub(super) stacked: StackedMoeBuffers,
    pub(super) shared_gate_proj: &'a Linear,
    pub(super) shared_up_proj: &'a Linear,
    pub(super) shared_down_proj: &'a Linear,
    pub(super) shared_inter_dim: usize,
}

impl MetalExecutor {
    #[expect(
        clippy::too_many_arguments,
        reason = "validation tail MoE: tenseurs, poids et top-k restent explicites"
    )]
    pub(super) fn check_tail_moe_shapes<'a>(
        &self,
        residual: &'a Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &'a Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
    ) -> Result<TailMoeShape<'a>> {
        ensure_biasless(o_proj, "o_proj")?;
        ensure_biasless(router, "router")?;
        let (batch, hidden_dim) = residual.as_matrix()?;
        let (context_batch, context_dim) = context.as_matrix()?;
        if batch != 1 || context_batch != 1 {
            return Err(InferError::Dimension(format!(
                "tail MoE Metal attend batch=1, reçu residual={batch}, context={context_batch}"
            )));
        }
        let o_out_dim = linear_out_dim(o_proj.weight())?;
        if o_out_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE o_proj sort {o_out_dim}, hidden={hidden_dim}"
            )));
        }
        let norm_weight = match post_norm.shape() {
            [dim] if *dim == hidden_dim => post_norm,
            [1, dim] if *dim == hidden_dim => post_norm,
            shape => {
                return Err(InferError::Dimension(format!(
                    "tail MoE norm attendu [{hidden_dim}], reçu {shape:?}"
                )))
            }
        };
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "tail MoE routeur experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if hidden_dim != stacked.gate.in_dim || hidden_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE hidden={hidden_dim}, gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "tail MoE inter dims gate={} up={} down_in={}",
                stacked.gate.out_dim, stacked.up.out_dim, stacked.down.in_dim
            )));
        }
        Ok(TailMoeShape {
            batch,
            hidden_dim,
            context_dim,
            norm_weight,
            expert_count,
            inter_dim: stacked.gate.out_dim,
            stacked,
        })
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "validation tail MoE shared: tenseurs, poids et top-k restent explicites"
    )]
    pub(super) fn check_tail_moe_shared_shapes<'a>(
        &self,
        residual: &'a Tensor,
        context: &Tensor,
        o_proj: &Linear,
        post_norm: &'a Tensor,
        router: &Linear,
        experts: &[GatedMlp],
        top_k: usize,
        shared_expert: &'a GatedMlp,
        shared_gate: &Linear,
    ) -> Result<TailMoeSharedShape<'a>> {
        ensure_biasless(o_proj, "o_proj")?;
        ensure_biasless(router, "router")?;
        ensure_biasless(shared_gate, "shared_gate")?;
        let (batch, hidden_dim) = residual.as_matrix()?;
        let (context_batch, context_dim) = context.as_matrix()?;
        if batch != 1 || context_batch != 1 {
            return Err(InferError::Dimension(format!(
                "tail MoE shared Metal attend batch=1, reçu residual={batch}, context={context_batch}"
            )));
        }
        let o_out_dim = linear_out_dim(o_proj.weight())?;
        if o_out_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE shared o_proj sort {o_out_dim}, hidden={hidden_dim}"
            )));
        }
        let norm_weight = match post_norm.shape() {
            [dim] if *dim == hidden_dim => post_norm,
            [1, dim] if *dim == hidden_dim => post_norm,
            shape => {
                return Err(InferError::Dimension(format!(
                    "tail MoE shared norm attendu [{hidden_dim}], reçu {shape:?}"
                )))
            }
        };
        let expert_count = linear_out_dim(router.weight())?;
        ensure_valid_top_k(top_k, expert_count)?;
        if expert_count != experts.len() {
            return Err(InferError::Dimension(format!(
                "tail MoE shared routeur experts={expert_count}, poids experts={}",
                experts.len()
            )));
        }
        let stacked = self.stacked_moe_buffers(experts)?;
        if hidden_dim != stacked.gate.in_dim || hidden_dim != stacked.up.in_dim {
            return Err(InferError::Dimension(format!(
                "tail MoE shared hidden={hidden_dim}, gate_in={}, up_in={}",
                stacked.gate.in_dim, stacked.up.in_dim
            )));
        }
        if stacked.gate.out_dim != stacked.up.out_dim || stacked.down.in_dim != stacked.gate.out_dim
        {
            return Err(InferError::Dimension(format!(
                "tail MoE shared inter dims gate={} up={} down_in={}",
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
                "tail shared gate sort {shared_gate_out_dim}, attendu 1"
            )));
        }
        let shared_inter_dim = linear_out_dim(shared_gate_proj.weight())?;
        let shared_up_dim = linear_out_dim(shared_up_proj.weight())?;
        let shared_down_dim = linear_out_dim(shared_down_proj.weight())?;
        let shared_down_in_dim = linear_in_dim(shared_down_proj.weight())?;
        if shared_inter_dim != shared_up_dim || shared_inter_dim != shared_down_in_dim {
            return Err(InferError::Dimension(format!(
                "tail shared expert dims gate={shared_inter_dim}, up={shared_up_dim}, down_in={shared_down_in_dim}"
            )));
        }
        if shared_down_dim != hidden_dim {
            return Err(InferError::Dimension(format!(
                "tail shared expert out={shared_down_dim}, hidden={hidden_dim}"
            )));
        }
        Ok(TailMoeSharedShape {
            batch,
            hidden_dim,
            context_dim,
            norm_weight,
            expert_count,
            inter_dim: stacked.gate.out_dim,
            stacked,
            shared_gate_proj,
            shared_up_proj,
            shared_down_proj,
            shared_inter_dim,
        })
    }
}
