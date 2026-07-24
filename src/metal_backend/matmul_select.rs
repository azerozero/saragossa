//! Sélection des chemins d'encodage matmul Metal.

use super::*;

#[derive(Clone, Copy)]
pub(super) enum AffineMatmulKernel {
    Qmm2,
    QmmNaFusedTiledU4,
    QmmNaFusedTiledU4Align64,
    FastQmvU4,
    FastQmvU4Align64,
    FastQmvU6,
    FastQmvOneU8,
    QmmNaFusedTiledU8,
    QmmNaU8Gs128,
    QmmNaU8,
    FastQmvU8,
    FastQmvU8Align64,
    Fallback,
}

impl MetalExecutor {
    #[inline]
    pub(super) fn select_resident_affine_matmul_kernel(
        &self,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
        group_size: usize,
        bits: usize,
        prefer_fast_affine: bool,
    ) -> AffineMatmulKernel {
        if can_use_fast_affine_qmm2_buffers(batch, in_dim, out_dim, group_size, bits)
            || can_use_fast_affine_qmm2_u8_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::Qmm2
        } else if self.qmm_na_fused_tiled_u4_available(group_size)
            && can_use_qmm_na_fused_tiled_u4_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::QmmNaFusedTiledU4
        } else if self.qmm_na_fused_tiled_u4_align64_available(group_size)
            && can_use_qmm_na_fused_tiled_u4_align64_buffers(
                batch, in_dim, out_dim, group_size, bits,
            )
        {
            AffineMatmulKernel::QmmNaFusedTiledU4Align64
        } else if (fast_affine_qmv_enabled(out_dim) || prefer_fast_affine)
            && batch > 0
            && bits == FAST_QMV_BITS
            && group_size == FAST_QMV_GROUP_SIZE
            && in_dim % 512 == 0
        {
            AffineMatmulKernel::FastQmvU4
        } else if can_use_fast_affine_qmv_u6_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvU6
        } else if can_use_fast_affine_qmv_one_u8_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvOneU8
        } else if self.qmm_na_fused_tiled_available(group_size)
            && can_use_qmm_na_fused_tiled_u8_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::QmmNaFusedTiledU8
        } else if self.na_gemm_coop_qb_gs128.is_some()
            && can_use_qmm_na_u8_gs128_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::QmmNaU8Gs128
        } else if can_use_fast_affine_qmv_u8_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvU8
        } else if can_use_fast_affine_qmv_u8_align64_buffers(
            batch, in_dim, out_dim, group_size, bits,
        ) {
            AffineMatmulKernel::FastQmvU8Align64
        } else if can_use_fast_affine_qmv_u4_align64_buffers(
            batch, in_dim, out_dim, group_size, bits,
        ) {
            AffineMatmulKernel::FastQmvU4Align64
        } else {
            AffineMatmulKernel::Fallback
        }
    }

    #[inline]
    pub(super) fn select_owned_affine_matmul_kernel(
        &self,
        batch: usize,
        in_dim: usize,
        weight: &AffineQuantizedTensor,
        prefer_fast_affine: bool,
    ) -> AffineMatmulKernel {
        if can_use_fast_affine_qmm2(batch, in_dim, weight)
            || can_use_fast_affine_qmm2_u8(batch, in_dim, weight)
        {
            AffineMatmulKernel::Qmm2
        } else if self.qmm_na_fused_tiled_u4_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u4(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaFusedTiledU4
        } else if self.qmm_na_fused_tiled_u4_align64_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u4_align64(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaFusedTiledU4Align64
        } else if can_use_fast_affine_qmv(batch, in_dim, weight)
            || (prefer_fast_affine && can_use_fast_affine_qmv_shape(batch, in_dim, weight))
        {
            AffineMatmulKernel::FastQmvU4
        } else if can_use_fast_affine_qmv_u6(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU6
        } else if can_use_fast_affine_qmv_one_u8(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvOneU8
        } else if self.qmm_na_fused_tiled_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u8(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaFusedTiledU8
        } else if self.na_gemm_coop_qb_gs128.is_some()
            && can_use_qmm_na_u8_gs128(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaU8Gs128
        } else if self.na_gemm_bf16.is_some() && can_use_qmm_na_u8(batch, in_dim, weight) {
            AffineMatmulKernel::QmmNaU8
        } else if can_use_fast_affine_qmv_u8(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU8
        } else if can_use_fast_affine_qmv_u8_align64(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU8Align64
        } else if can_use_fast_affine_qmv_u4_align64(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU4Align64
        } else {
            AffineMatmulKernel::Fallback
        }
    }
}
