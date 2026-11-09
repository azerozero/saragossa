//! Sélection des chemins d'encodage matmul Metal.

use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AffineMatmulKernel {
    Qmm2,
    QmvT,
    Skinny,
    Qmm8,
    QmmNaFusedTiledU4,
    QmmNaFusedTiledU4Align64,
    FastQmvU4,
    FastQmvU4Align64,
    FastQmvU2,
    FastQmvU3,
    FastQmvU6,
    FastQmvOneU8,
    QmmNaFusedTiledU8,
    QmmNaU8Gs128,
    QmmNaU8,
    FastQmvU8,
    FastQmvU8Align64,
    Fallback,
}

impl AffineMatmulKernel {
    /// Les offsets Metal sont en octets. Seul le chargeur NA tuilé u4
    /// (`load_b_tile_quant_u4_gs64_split`, x3 et align64 compris) caste packed
    /// en uint4 : 16 octets. Les autres lisent des uint/ushort et les affines
    /// des bfloat scalaires : contrats des arguments uint* (4) et bfloat* (2).
    /// « align64 » décrit K % 64, pas une adresse alignée sur 64 octets.
    pub(super) fn supports_offsets(self, packed: u64, scales: u64, biases: u64) -> bool {
        let packed_alignment = match self {
            Self::QmmNaFusedTiledU4 | Self::QmmNaFusedTiledU4Align64 => 16,
            _ => 4,
        };
        packed % packed_alignment == 0 && scales % 2 == 0 && biases % 2 == 0
    }

    pub(super) fn with_offsets(self, packed: u64, scales: u64, biases: u64) -> Self {
        if self.supports_offsets(packed, scales, biases) {
            self
        } else {
            Self::Fallback
        }
    }
}

#[cfg(test)]
thread_local! {
    // Exerce chaque encodeur même si le matériel ou les flags privilégient
    // un autre chemin. Le contrôle d'alignement de production reste appliqué.
    pub(super) static OFFSET_TEST_KERNEL: Cell<Option<AffineMatmulKernel>> = const { Cell::new(None) };
}

impl MetalExecutor {
    /// Abaisse le plancher du GEMM NA u4 aux petites projections de flint,
    /// servies par le bf16×3 (`gemm_nax_coop_qb_tiled_u4_x3`). Appelé à
    /// l'installation de l'executor sur un décodeur flint seulement : Qwen et
    /// Gemma gardent le plancher 2048 et restent bit-identiques. Sans le
    /// pipeline x3, le plancher reste en place (repli qmv).
    pub(crate) fn enable_flint_small_nax_gemm(&mut self) {
        if self.na_gemm_coop_qb_tiled_u4_x3.is_some() {
            self.qmm_na_u4_min_dim = QMM_NA_U4_FLINT_MIN_DIM;
        }
    }

    /// Ouvre ce seul poids au GEMM NA u4 sous le plancher, servi par le
    /// bf16×3 : appelé pour les k/v des couches full du modèle cible (Qwen
    /// dense à k/v 5120→1024). L'autorisation suit l'identité du poids
    /// (`weight_id`, stable quand son stockage passe en mémoire partagée) et
    /// non sa forme : un drafter DFlash ou une tête MTP de même forme garde le
    /// qmv. Seul le sélecteur owned la consulte, le chemin du prefill cible ;
    /// le sélecteur résident (buffers sans identité de poids) garde le
    /// plancher. Refusé hors u4 gs64, au-dessus du plancher ou sans pipeline
    /// x3 (repli qmv).
    pub(crate) fn allow_qmm_na_u4_x3_weight(&mut self, weight: &AffineQuantizedTensor) {
        let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
            return;
        };
        let in_dim = match weight.shape() {
            [_, in_dim] => *in_dim,
            _ => return,
        };
        if self.na_gemm_coop_qb_tiled_u4_x3.is_some()
            && weight.bits() == 4
            && weight.group_size() == FAST_QMV_GROUP_SIZE
            && in_dim.min(out_dim) < QMM_NA_U4_DEFAULT_MIN_DIM
            && !self.qmm_na_u4_x3_weights.contains(&weight.weight_id())
        {
            self.qmm_na_u4_x3_weights.push(weight.weight_id());
        }
    }

    /// Plancher du GEMM NA u4 pour `weight` : nul pour un poids ouvert par
    /// [`Self::allow_qmm_na_u4_x3_weight`], sinon le plancher de l'executor.
    /// La surcharge de test l'emporte sur les deux.
    #[inline]
    fn qmm_na_u4_min_dim_for(&self, weight: &AffineQuantizedTensor) -> usize {
        #[cfg(test)]
        if let Some(min_dim) = super::flags::qmm_na_u4_min_dim_test_override() {
            return min_dim;
        }
        if self.qmm_na_u4_x3_weights.contains(&weight.weight_id()) {
            0
        } else {
            self.qmm_na_u4_min_dim
        }
    }

    /// Plancher effectif du GEMM NA u4 (surcharge de test comprise).
    #[inline]
    pub(super) fn qmm_na_u4_min_dim(&self) -> usize {
        #[cfg(test)]
        if let Some(min_dim) = super::flags::qmm_na_u4_min_dim_test_override() {
            return min_dim;
        }
        self.qmm_na_u4_min_dim
    }

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
        #[cfg(test)]
        if let Some(kernel) = OFFSET_TEST_KERNEL.with(Cell::get) {
            return kernel;
        }
        // Vérification greedy stricte : seuls les chemins exacts en lignes (qmm2,
        // qmv-T, qmv répliqué) ; les GEMM NA (MMA bf16) sont exclus à toute largeur.
        let strict = super::verify_strict_active();
        if can_use_fast_affine_qmm2_buffers(batch, in_dim, out_dim, group_size, bits)
            || can_use_fast_affine_qmm2_u3_buffers(batch, in_dim, out_dim, group_size, bits)
            || can_use_fast_affine_qmm2_u8_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::Qmm2
        } else if can_use_qmvt_buffers(batch, in_dim, out_dim, group_size, bits, prefer_fast_affine)
        {
            AffineMatmulKernel::QmvT
        } else if self.qmm_skinny_available()
            && can_use_qmm_skinny_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::Skinny
        } else if can_use_fast_affine_qmm8_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::Qmm8
        } else if !strict
            && self.qmm_na_fused_tiled_u4_available(group_size)
            && can_use_qmm_na_fused_tiled_u4_buffers(
                batch,
                in_dim,
                out_dim,
                group_size,
                bits,
                self.qmm_na_u4_min_dim(),
            )
        {
            AffineMatmulKernel::QmmNaFusedTiledU4
        } else if !strict
            && self.qmm_na_fused_tiled_u4_align64_available(group_size)
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
        } else if can_use_fast_affine_qmv_u2_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvU2
        } else if can_use_fast_affine_qmv_u3_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvU3
        } else if can_use_fast_affine_qmv_u6_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvU6
        } else if can_use_fast_affine_qmv_one_u8_buffers(batch, in_dim, out_dim, group_size, bits) {
            AffineMatmulKernel::FastQmvOneU8
        } else if !strict
            && self.qmm_na_fused_tiled_available(group_size)
            && can_use_qmm_na_fused_tiled_u8_buffers(batch, in_dim, out_dim, group_size, bits)
        {
            AffineMatmulKernel::QmmNaFusedTiledU8
        } else if !strict
            && self.na_gemm_coop_qb_gs128.is_some()
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
        #[cfg(test)]
        if let Some(kernel) = OFFSET_TEST_KERNEL.with(Cell::get) {
            return kernel;
        }
        // Vérification greedy stricte : seuls les chemins exacts en lignes (qmm2,
        // qmv-T, qmv répliqué) ; les GEMM NA (MMA bf16) sont exclus à toute largeur.
        let strict = super::verify_strict_active();
        if can_use_fast_affine_qmm2(batch, in_dim, weight)
            || can_use_fast_affine_qmm2_u3(batch, in_dim, weight)
            || can_use_fast_affine_qmm2_u8(batch, in_dim, weight)
        {
            AffineMatmulKernel::Qmm2
        } else if can_use_qmvt(batch, in_dim, weight, prefer_fast_affine) {
            AffineMatmulKernel::QmvT
        } else if self.qmm_skinny_available() && can_use_qmm_skinny(batch, in_dim, weight) {
            AffineMatmulKernel::Skinny
        } else if can_use_fast_affine_qmm8(batch, in_dim, weight) {
            AffineMatmulKernel::Qmm8
        } else if !strict
            && self.qmm_na_fused_tiled_u4_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u4(
                batch,
                in_dim,
                weight,
                self.qmm_na_u4_min_dim_for(weight),
            )
        {
            AffineMatmulKernel::QmmNaFusedTiledU4
        } else if !strict
            && self.qmm_na_fused_tiled_u4_align64_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u4_align64(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaFusedTiledU4Align64
        } else if can_use_fast_affine_qmv(batch, in_dim, weight)
            || (prefer_fast_affine && can_use_fast_affine_qmv_shape(batch, in_dim, weight))
        {
            AffineMatmulKernel::FastQmvU4
        } else if can_use_fast_affine_qmv_u2(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU2
        } else if can_use_fast_affine_qmv_u3(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU3
        } else if can_use_fast_affine_qmv_u6(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvU6
        } else if can_use_fast_affine_qmv_one_u8(batch, in_dim, weight) {
            AffineMatmulKernel::FastQmvOneU8
        } else if !strict
            && self.qmm_na_fused_tiled_available(weight.group_size())
            && can_use_qmm_na_fused_tiled_u8(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaFusedTiledU8
        } else if !strict
            && self.na_gemm_coop_qb_gs128.is_some()
            && can_use_qmm_na_u8_gs128(batch, in_dim, weight)
        {
            AffineMatmulKernel::QmmNaU8Gs128
        } else if !strict && self.na_gemm_bf16.is_some() && can_use_qmm_na_u8(batch, in_dim, weight)
        {
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
