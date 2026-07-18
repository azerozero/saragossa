//! Allocation des scratch GPU des chemins MoE shared et routed.

use super::*;

impl MetalExecutor {
    /// Alloue (ou récupère mémoïsés) les scratch du MoE routed-only.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(super) fn allocate_moe_routed_scratch(
        &self,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
    ) -> Result<MoeRoutedScratch> {
        Ok(MoeRoutedScratch {
            router: self.private_f32_buffer(expert_count, "moe_routed_router_logits")?,
            indices: self.private_u32_buffer(top_k, "moe_routed_indices")?,
            scores: self.private_f32_buffer(top_k, "moe_routed_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed gate")?,
                "moe_routed_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed up")?,
                "moe_routed_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe routed hidden")?,
                "moe_routed_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, out_dim, "moe routed down")?,
                "moe_routed_down",
            )?,
        })
    }

    /// Alloue les scratch du MoE routed-only batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(super) fn allocate_moe_routed_rows_scratch(
        &self,
        rows: usize,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
    ) -> Result<MoeRoutedRowsScratch> {
        let total_topk = checked_len(rows, top_k, "moe routed rows topk total")?;
        Ok(MoeRoutedRowsScratch {
            router: self.private_f32_buffer(
                checked_len(rows, expert_count, "moe routed rows router")?,
                "moe_routed_rows_router_logits",
            )?,
            indices: self.private_u32_buffer(total_topk, "moe_routed_rows_indices")?,
            scores: self.private_f32_buffer(total_topk, "moe_routed_rows_scores")?,
            gate: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe routed rows gate")?,
                "moe_routed_rows_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe routed rows up")?,
                "moe_routed_rows_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe routed rows hidden")?,
                "moe_routed_rows_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(total_topk, out_dim, "moe routed rows down")?,
                "moe_routed_rows_down",
            )?,
        })
    }

    /// Alloue (ou récupère mémoïsés) les scratch du MoE shared mono-token.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(in super::super) fn allocate_moe_shared_scratch(
        &self,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
        shared_inter_dim: usize,
    ) -> Result<MoeSharedScratch> {
        Ok(MoeSharedScratch {
            router: self.private_f32_buffer(expert_count, "moe_shared_router_logits")?,
            indices: self.private_u32_buffer(top_k, "moe_shared_indices")?,
            scores: self.private_f32_buffer(top_k, "moe_shared_scores")?,
            gate: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared gate")?,
                "moe_shared_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared up")?,
                "moe_shared_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(top_k, inter_dim, "moe shared hidden")?,
                "moe_shared_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(top_k, out_dim, "moe shared down")?,
                "moe_shared_down",
            )?,
            shared_gate: self.private_f32_buffer(1, "moe_shared_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_gate")?,
            shared_up: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_up")?,
            shared_hidden: self.private_f32_buffer(shared_inter_dim, "moe_shared_proj_hidden")?,
            shared_down: self.private_f32_buffer(out_dim, "moe_shared_proj_down")?,
        })
    }

    /// Alloue (ou récupère mémoïsés) les scratch du MoE shared batché.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une taille déborde ou si l'allocation Metal échoue.
    pub(in super::super) fn allocate_moe_shared_rows_scratch(
        &self,
        rows: usize,
        top_k: usize,
        expert_count: usize,
        inter_dim: usize,
        out_dim: usize,
        shared_inter_dim: usize,
    ) -> Result<MoeSharedRowsScratch> {
        let total_topk = checked_len(rows, top_k, "moe shared rows topk total")?;
        Ok(MoeSharedRowsScratch {
            router: self.private_f32_buffer(
                checked_len(rows, expert_count, "moe shared rows router")?,
                "moe_shared_rows_router_logits",
            )?,
            indices: self.private_u32_buffer(total_topk, "moe_shared_rows_indices")?,
            scores: self.private_f32_buffer(total_topk, "moe_shared_rows_scores")?,
            gate: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows gate")?,
                "moe_shared_rows_gate",
            )?,
            up: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows up")?,
                "moe_shared_rows_up",
            )?,
            hidden: self.private_f32_buffer(
                checked_len(total_topk, inter_dim, "moe shared rows hidden")?,
                "moe_shared_rows_hidden",
            )?,
            down: self.private_f32_buffer(
                checked_len(total_topk, out_dim, "moe shared rows down")?,
                "moe_shared_rows_down",
            )?,
            shared_gate: self.private_f32_buffer(rows, "moe_shared_rows_gate_scalar")?,
            shared_proj_gate: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj gate")?,
                "moe_shared_rows_proj_gate",
            )?,
            shared_up: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj up")?,
                "moe_shared_rows_proj_up",
            )?,
            shared_hidden: self.private_f32_buffer(
                checked_len(rows, shared_inter_dim, "moe shared rows proj hidden")?,
                "moe_shared_rows_proj_hidden",
            )?,
            shared_down: self.private_f32_buffer(
                checked_len(rows, out_dim, "moe shared rows proj down")?,
                "moe_shared_rows_proj_down",
            )?,
        })
    }
}
