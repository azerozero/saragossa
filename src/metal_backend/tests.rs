pub(crate) mod oracle;

use super::matmul_select::AffineMatmulKernel;
use super::*;

include!("tests/util.rs");
include!("tests/buffers_core.rs");
include!("tests/gdn_weight_storage.rs");
include!("tests/affine_offsets.rs");
mod dense_bf16;
include!("tests/na_gemm_delta.rs");
include!("tests/small_m_qmm.rs");
include!("tests/coop_moe.rs");
include!("tests/quant_qmv_sampling.rs");
include!("tests/qmv_u5.rs");
include!("tests/qmv_align64.rs");
include!("tests/moe_duo.rs");
include!("tests/linear_attention.rs");
include!("tests/attention.rs");
include!("tests/mtp_seed.rs");
include!("tests/verify_tree.rs");
include!("tests/moe_verify_rows.rs");
include!("tests/rows_exact.rs");
include!("tests/flash_next.rs");
