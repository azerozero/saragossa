//! Prédicats d'éligibilité des chemins rapides Metal.

use super::*;

pub(super) fn ensure_biasless(linear: &Linear, label: &'static str) -> Result<()> {
    if linear.bias().is_some() {
        return Err(InferError::Config(format!(
            "MoE Metal ne supporte pas les biais expert {label}"
        )));
    }
    Ok(())
}

pub(super) fn dense_vector<'a>(
    tensor: &'a Tensor,
    dim: usize,
    label: &'static str,
) -> Result<&'a [f32]> {
    match tensor.shape() {
        [n] if *n == dim => Ok(tensor.data()),
        [1, n] if *n == dim => Ok(tensor.data()),
        shape => Err(InferError::Dimension(format!(
            "{label} attendu [{dim}] ou [1,{dim}], reçu {shape:?}"
        ))),
    }
}

pub(super) fn linear_out_dim(weight: &LinearWeight) -> Result<usize> {
    match weight.shape() {
        [out_dim, _] => Ok(*out_dim),
        shape => Err(InferError::Dimension(format!(
            "poids Linear attendu rang 2, reçu {shape:?}"
        ))),
    }
}

pub(super) fn linear_in_dim(weight: &LinearWeight) -> Result<usize> {
    match weight.shape() {
        [_, in_dim] => Ok(*in_dim),
        shape => Err(InferError::Dimension(format!(
            "poids Linear attendu rang 2, reçu {shape:?}"
        ))),
    }
}

pub(super) fn expect_linear_shape(
    weight: &LinearWeight,
    expected_out: usize,
    expected_in: usize,
    label: &'static str,
) -> Result<()> {
    match weight.shape() {
        [out_dim, in_dim] if *out_dim == expected_out && *in_dim == expected_in => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{label}.weight attendu [{expected_out},{expected_in}], reçu {shape:?}"
        ))),
    }
}

pub(super) fn expect_linear_in(
    weight: &LinearWeight,
    expected_in: usize,
    label: &'static str,
) -> Result<()> {
    match weight.shape() {
        [_, in_dim] if *in_dim == expected_in => Ok(()),
        shape => Err(InferError::Dimension(format!(
            "{label}.weight entrée attendue {expected_in}, reçu {shape:?}"
        ))),
    }
}

pub(super) fn can_use_fast_affine_qmv(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    fast_affine_qmv_enabled(out_dim) && can_use_fast_affine_qmv_shape(batch, in_dim, weight)
}

pub(super) fn can_use_fast_affine_qmv_shape(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    batch > 0
        && weight.bits() == FAST_QMV_BITS
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_u6(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u6_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmm2_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u8_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Vérifie l'éligibilité d'un poids u8 gs64 au qmv aligné 64.
pub(super) fn can_use_fast_affine_qmv_u8_align64(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u8_align64_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Vérifie l'éligibilité d'un poids u4 gs64 au qmv aligné 64.
pub(super) fn can_use_fast_affine_qmv_u4_align64(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_u4_align64_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Prédicat du GEMM NA tuilé à dé-quant fusionnée : A bf16, B u8 staged en
/// threadgroup bf16 par tuile BM=BN=BK=64. Opt-in, car le gain dépend du trafic
/// poids et les résultats suivent l'ordre d'accumulation tuilé.
pub(super) fn can_use_qmm_na_fused_tiled_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_fused_tiled_enabled()
        && batch >= 16
        && out_dim % 64 == 0
        && bits == 8
        && matches!(group_size, FAST_QMV_GROUP_SIZE | QMM_NA_GS128_GROUP_SIZE)
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_fused_tiled_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_fused_tiled_u8_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Variante 4-bit gs64 du GEMM NA tuilé. Le chemin reste borné aux grandes
/// projections denses : les petites projections shared-expert divergent en
/// oracle greedy avec l'accumulation bf16 tensor-core et gardent le qmv f32.
pub(super) fn can_use_qmm_na_fused_tiled_u4_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_fused_tiled_enabled()
        && qmm_na_fused_tiled_u4_enabled()
        && batch >= 16
        && out_dim % 64 == 0
        && out_dim >= 2048
        && in_dim >= 2048
        && bits == 4
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_fused_tiled_u4(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_fused_tiled_u4_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Vérifie l'éligibilité du GEMM NA u4 pour K aligné sur 64 hors chemin Qwen.
pub(super) fn can_use_qmm_na_fused_tiled_u4_align64_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_fused_tiled_enabled()
        && qmm_na_fused_tiled_u4_enabled()
        && batch >= 16
        && out_dim % 64 == 0
        && out_dim >= 2048
        && in_dim >= 2048
        && bits == 4
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 64 == 0
        && in_dim % 512 != 0
}

/// Vérifie l'éligibilité owned du GEMM NA u4 aligné sur 64.
pub(super) fn can_use_qmm_na_fused_tiled_u4_align64(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_fused_tiled_u4_align64_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Prédicat du GEMM prefill sur Neural Accelerators (matmul2d bf16) : dé-quant
/// u8→bf16 transposée du poids + activations bf16 + tensor-cores. `batch` grand
/// (prefill). Opt-in (`RETI_RUST_QMM_NA`) ; l'appelant vérifie EN PLUS que la NA est
/// dispo (`na_gemm_bf16.is_some()`, macOS≥26). bf16 ⇒ non bit-à-bit identique.
pub(super) fn can_use_qmm_na_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    qmm_na_enabled()
        && batch >= 16
        && weight.bits() == 8
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

/// Prédicat du GEMM NA fusé dense gs128 : dé-quant u8 directement dans le kernel.
/// Le kernel masque la queue M ; N reste aligné 32 pour éviter les stores hors poids.
pub(super) fn can_use_qmm_na_u8_gs128_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmm_na_gs128_enabled()
        && batch >= 16
        && out_dim % 32 == 0
        && bits == 8
        && group_size == QMM_NA_GS128_GROUP_SIZE
        && in_dim % 512 == 0
}

pub(super) fn can_use_qmm_na_u8_gs128(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_qmm_na_u8_gs128_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmv_one_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmv_one_u8_buffers(
        batch,
        in_dim,
        out_dim,
        weight.group_size(),
        weight.bits(),
    ) && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2_u8(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    let Some(out_dim) = fast_affine_qmv_out_dim(weight) else {
        return false;
    };
    can_use_fast_affine_qmm2_u8_buffers(batch, in_dim, out_dim, weight.group_size(), weight.bits())
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn can_use_fast_affine_qmm2_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch == 2
        && bits == FAST_QMV_BITS
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && out_dim % 8 == 0
}

/// Prédicat du qmv 6-bit gs64 pour le talker TTS : même contrat de buffers que
/// les qmv rapides u4/u8, mais dépaquetage 6-bit identique au kernel générique.
pub(super) fn can_use_fast_affine_qmv_u6_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmv_u6_enabled()
        && fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == FAST_QMV_U6_BITS
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % FAST_QMV_GROUP_SIZE == 0
}

/// Prédicat du qmv 8-bit aligné : même géométrie que le qmv 4-bit rapide,
/// mais poids oQ/DWQ en u8.
pub(super) fn can_use_fast_affine_qmv_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == 8
        && matches!(group_size, FAST_QMV_GROUP_SIZE | 128)
        && in_dim % 512 == 0
        && out_dim % 8 == 0
}

/// Vérifie le qmv u8 gs64 réservé aux entrées alignées 64, mais pas 512.
pub(super) fn can_use_fast_affine_qmv_u8_align64_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == 8
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % FAST_QMV_GROUP_SIZE == 0
        && in_dim % 512 != 0
        && out_dim % 8 == 0
}

/// Vérifie le qmv u4 gs64 réservé aux entrées alignées 64, mais pas 512.
pub(super) fn can_use_fast_affine_qmv_u4_align64_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    fast_affine_qmv_enabled(out_dim)
        && batch > 0
        && bits == FAST_QMV_BITS
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % FAST_QMV_GROUP_SIZE == 0
        && in_dim % 512 != 0
        && out_dim % 8 == 0
}

/// Prédicat du qmv scalaire 8-bit gs64 : cible `shared_expert_gate` oQ
/// (out_dim=1), actif par défaut après A/B et désactivable par env.
pub(super) fn can_use_fast_affine_qmv_one_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    qmv_one_u8_enabled()
        && batch > 0
        && out_dim == 1
        && bits == 8
        && group_size == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
}

/// Prédicat du qmm2 8-bit (duo light-batch sur poids DWQ) : mêmes gates que le
/// qmv u8 aligned, à batch == 2.
pub(super) fn can_use_fast_affine_qmm2_u8_buffers(
    batch: usize,
    in_dim: usize,
    out_dim: usize,
    group_size: usize,
    bits: usize,
) -> bool {
    batch == 2 && can_use_fast_affine_qmv_u8_buffers(batch, in_dim, out_dim, group_size, bits)
}

pub(super) fn can_use_fast_affine_argmax_qmv(
    batch: usize,
    in_dim: usize,
    weight: &AffineQuantizedTensor,
) -> bool {
    fast_argmax_qmv_enabled()
        && batch == 1
        && weight.bits() == FAST_QMV_BITS
        && weight.group_size() == FAST_QMV_GROUP_SIZE
        && in_dim % 512 == 0
        && matches!(weight.shape(), [_, weight_in_dim] if *weight_in_dim == in_dim)
}

pub(super) fn fast_affine_qmv_out_dim(weight: &AffineQuantizedTensor) -> Option<usize> {
    match weight.shape() {
        [out_dim, _] => Some(*out_dim),
        _ => None,
    }
}

pub(super) fn can_use_dense_qmv_fast(batch: usize, in_dim: usize, out_dim: usize) -> bool {
    dense_qmv_fast_enabled() && batch > 0 && in_dim % 512 == 0 && out_dim % 8 == 0
}

pub(super) fn can_use_fast_gather_qmv(lhs_rows: usize, weight: &StackedAffineBuffers) -> bool {
    fast_gather_qmv_enabled(weight)
        && lhs_rows > 0
        && ((weight.bits == FAST_QMV_BITS
            && weight.group_size == FAST_QMV_GROUP_SIZE
            && weight.in_dim % weight.group_size == 0)
            || (weight.bits == 8
                && matches!(weight.group_size, FAST_QMV_GROUP_SIZE | 128)
                && weight.in_dim % 512 == 0
                && weight.out_dim % 8 == 0))
}

pub(super) fn valid_gather_lhs_rows(lhs_rows: usize, topk: usize) -> bool {
    lhs_rows > 0 && (lhs_rows == 1 || lhs_rows == topk || topk % lhs_rows == 0)
}

pub(super) fn can_use_fast_gather_pair_qmv(
    lhs_rows: usize,
    gate: &StackedAffineBuffers,
    up: &StackedAffineBuffers,
) -> bool {
    fused_gate_up_enabled()
        && (gate.bits != 8 || fused_gate_up_u8_enabled())
        && can_use_fast_gather_qmv(lhs_rows, gate)
        && can_use_fast_gather_qmv(lhs_rows, up)
        && gate.experts == up.experts
        && gate.out_dim == up.out_dim
        && gate.in_dim == up.in_dim
        && gate.packed_cols == up.packed_cols
        && gate.group_size == up.group_size
        && gate.bits == up.bits
        && gate.groups == up.groups
}

pub(super) fn can_fuse_shared_gate_up_weights(gate: &Linear, up: &Linear) -> bool {
    if !fused_shared_gate_up_enabled() {
        return false;
    }
    match (gate.weight(), up.weight()) {
        (LinearWeight::AffineQuantized(gate), LinearWeight::AffineQuantized(up))
            if gate.bits() == 8 || up.bits() == 8 =>
        {
            fused_shared_gate_up_u8_enabled()
        }
        _ => true,
    }
}

pub(super) fn can_fuse_shared_gate_up_buffers(
    gate: &MetalLinearWeightBuffers,
    up: &MetalLinearWeightBuffers,
) -> bool {
    if !fused_shared_gate_up_enabled() {
        return false;
    }
    match (gate, up) {
        (
            MetalLinearWeightBuffers::AffineQuantized {
                bits: gate_bits, ..
            },
            MetalLinearWeightBuffers::AffineQuantized { bits: up_bits, .. },
        ) if *gate_bits == 8 || *up_bits == 8 => fused_shared_gate_up_u8_enabled(),
        _ => true,
    }
}
