//! Préfixe de l'attention en lignes de la vérification sur le NAX du prefill.
//!
//! Les `gqa × lignes` requêtes d'une tête KV forment une tuile Q packée
//! (`[kv_head][ligne·gqa + tête][d]`, bf16) : le NAX tourne avec `H = kv_heads`
//! et `gqa_factor = 1`, sans causalité (le préfixe précède toutes les lignes).
//! La dimension batch sert de split-K : le batch `z` lit le bloc de clés
//! `[z·kL, (z+1)·kL)` et écrit sa sortie normalisée et son LSE (log naturel)
//! dans la tranche `z` des partiels, que l'appelant fusionne.

use super::attention::{checked_i32, checked_i64, SteelAttnParams, NAX_CAUSAL_ATTN_BQ64};
use super::byte_helpers::checked_len;
use super::pipelines::{compile_nax_function, steel_attention_nax_source};
use super::*;

/// Instanciation d128 (noyau plein, WM=4, WN=1), la seule compilée par MLX.
const NAX_ROWS_D128: &str = "steel_attention_nax_bf16_f32out_bq64_bk32_bd128_wm4_wn1";
/// Tuile Q et tuile K des deux instanciations retenues.
const NAX_ROWS_BQ: usize = 64;
const NAX_ROWS_BK: usize = 32;

/// Pipeline NAX non causal avec LSE, pour une `head_dim` (128 ou 256).
#[derive(Debug)]
pub(crate) struct NaxRowsAttention {
    pipeline: ComputePipelineState,
    head_dim: usize,
    simdgroups: NSUInteger,
}

/// Géométrie d'un dispatch : `batches` blocs de `block_len` clés consécutives
/// à partir de la liaison de K/V, `q_rows = gqa × lignes` requêtes par tête KV.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NaxRowsDispatch {
    pub(crate) kv_heads: usize,
    pub(crate) q_rows: usize,
    pub(crate) block_len: usize,
    pub(crate) batches: usize,
}

/// Buffers liés d'un dispatch : `(buffer, offset en octets)`.
pub(crate) struct NaxRowsBuffers<'a> {
    pub(crate) q: &'a BufferRef,
    pub(crate) keys: (&'a BufferRef, u64),
    pub(crate) values: (&'a BufferRef, u64),
    pub(crate) out: (&'a BufferRef, u64),
    pub(crate) lse: (&'a BufferRef, u64),
}

impl NaxRowsAttention {
    /// Compile le NAX non causal avec LSE (constante 303) pour `head_dim`.
    /// `None` si la forme n'est pas couverte, si MetalPerformancePrimitives
    /// manque (macOS < 26) ou sans accélérateur : l'appelant garde son repli.
    pub(crate) fn compile(device: &Device, head_dim: usize) -> Option<Self> {
        let (name, simdgroups) = match head_dim {
            256 => (
                NAX_CAUSAL_ATTN_BQ64.host_name,
                NAX_CAUSAL_ATTN_BQ64.simdgroups,
            ),
            128 => (NAX_ROWS_D128, 4),
            _ => return None,
        };
        let pipeline = compile_nax_function(
            device,
            &steel_attention_nax_source(),
            name,
            &[
                (200, false),
                (201, false),
                (300, false),
                (301, false),
                (302, false),
                (303, true),
                (304, false),
                (305, false),
            ],
        )?;
        Some(Self {
            pipeline,
            head_dim,
            simdgroups,
        })
    }

    /// Encode un dispatch : sortie `[batch][kv_head][q_row][d]` (f32, normalisée)
    /// et LSE `[batch][kv_head][q_row]`, relatifs aux offsets de `out` et `lse`.
    /// K/V bf16 `[position][kv_head][d]` ; Q bf16 packé `[kv_head][q_row][d]`.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension est nulle ou déborde `i32`/`i64`.
    pub(crate) fn encode(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buffers: &NaxRowsBuffers<'_>,
        dispatch: NaxRowsDispatch,
    ) -> Result<()> {
        let NaxRowsDispatch {
            kv_heads,
            q_rows,
            block_len,
            batches,
        } = dispatch;
        if kv_heads == 0 || q_rows == 0 || block_len == 0 || batches == 0 {
            return Err(InferError::Dimension(format!(
                "nax rows: dimension nulle ({dispatch:?})"
            )));
        }
        let d = self.head_dim;
        let label = "nax rows";
        let kv_dim = checked_len(kv_heads, d, label)?;
        let q_head_stride = checked_len(q_rows, d, label)?;
        let out_batch = checked_len(kv_heads, q_head_stride, label)?;
        let kv_batch = checked_len(block_len, kv_dim, label)?;
        let n_q_aligned = q_rows / NAX_ROWS_BQ;
        let n_k_aligned = block_len / NAX_ROWS_BK;
        let kv_strides = [
            checked_i64(kv_batch)?,
            checked_i64(d)?,
            checked_i64(kv_dim)?,
        ];
        let params = SteelAttnParams {
            b: checked_i32(batches, label)?,
            h: checked_i32(kv_heads, label)?,
            d: checked_i32(d, label)?,
            q_l: checked_i32(q_rows, label)?,
            k_l: checked_i32(block_len, label)?,
            gqa_factor: 1,
            scale: (d as f32).sqrt().recip(),
            n_q: checked_i32(q_rows.div_ceil(NAX_ROWS_BQ), label)?,
            n_k: checked_i32(block_len.div_ceil(NAX_ROWS_BK), label)?,
            n_q_aligned: checked_i32(n_q_aligned, label)?,
            n_k_aligned: checked_i32(n_k_aligned, label)?,
            q_l_rem: checked_i32(q_rows - n_q_aligned * NAX_ROWS_BQ, label)?,
            k_l_rem: checked_i32(block_len - n_k_aligned * NAX_ROWS_BK, label)?,
            q_l_off: 0,
            // Batch = bloc split-K : toutes les tranches lisent le même Q.
            q_strides: [0, checked_i64(q_head_stride)?, checked_i64(d)?],
            k_strides: kv_strides,
            v_strides: kv_strides,
            o_strides: [
                checked_i64(out_batch)?,
                checked_i64(q_head_stride)?,
                checked_i64(d)?,
            ],
        };
        encoder.set_compute_pipeline_state(&self.pipeline);
        encoder.set_buffer(0, Some(buffers.q), 0);
        encoder.set_buffer(1, Some(buffers.keys.0), buffers.keys.1);
        encoder.set_buffer(2, Some(buffers.values.0), buffers.values.1);
        encoder.set_buffer(3, Some(buffers.out.0), buffers.out.1);
        encoder.set_bytes(
            4,
            std::mem::size_of::<SteelAttnParams>() as NSUInteger,
            std::ptr::from_ref(&params).cast::<c_void>(),
        );
        encoder.set_buffer(8, Some(buffers.lse.0), buffers.lse.1);
        let nsuint = |value: usize| {
            NSUInteger::try_from(value)
                .map_err(|_| InferError::Dimension(format!("{label}: {value} hors NSUInteger")))
        };
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new(
                nsuint(q_rows.div_ceil(NAX_ROWS_BQ))?,
                nsuint(kv_heads)?,
                nsuint(batches)?,
            ),
            MTLSize::new(32, self.simdgroups, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }
}
