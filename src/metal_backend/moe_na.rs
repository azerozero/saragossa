//! MoE routé prefill via le kernel `gather_qmm` porté (cooperative_tensor quantifié
//! groupé `gemm_nax_coop_qb_grouped`, ~7,6× le qmv-gather). Tri+pad des tokens par
//! expert (16-aligné, CPU à partir d'un readback des indices), UN dispatch par
//! projection lisant le poids EMPILÉ packed/scales/biases DIRECTEMENT (zéro
//! matérialisation bf16). Opt-in `SARAGOSSA_RUST_MOE_COOP`.

use super::*;

/// Colonnes N (et pas de K) d'une tuile des kernels groupés tuilés.
const COOP_TILED_BN: usize = 64;

#[cfg(test)]
thread_local! {
    /// Appels du MoE routé coop servis par les kernels tuilés sur ce thread.
    static MOE_COOP_TILED_TRACE: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Compteur des appels du MoE routé coop servis par les kernels tuilés sur ce
/// thread : prouve le chemin pris.
#[cfg(test)]
pub(crate) fn moe_coop_tiled_trace_for_test() -> usize {
    MOE_COOP_TILED_TRACE.with(std::cell::Cell::get)
}

/// Grille des kernels groupés : (tuiles M, tuiles N).
fn coop_grid(max_m: usize, n: usize, tiled: bool) -> MTLSize {
    if tiled {
        MTLSize::new(
            (max_m / MOE_COOP_TILED_BM) as u64,
            n.div_ceil(COOP_TILED_BN) as u64,
            1,
        )
    } else {
        MTLSize::new((max_m / MOE_COOP_TILE_M) as u64, n.div_ceil(32) as u64, 1)
    }
}

/// Threads par threadgroup : un simdgroup (16 × 32) ou quatre (tuilé, 2 × 2).
fn coop_threads(tiled: bool) -> MTLSize {
    MTLSize::new(if tiled { 128 } else { 32 }, 1, 1)
}

/// Préconditions des kernels tuilés, sans masque : u4, M en tuiles entières
/// de [`MOE_COOP_TILED_BM`], N et K multiples de 64.
fn check_coop_tiled_shape(bits: usize, max_m: usize, n: usize, k: usize) -> Result<()> {
    if bits != FAST_QMV_BITS
        || max_m % MOE_COOP_TILED_BM != 0
        || n % COOP_TILED_BN != 0
        || k % COOP_TILED_BN != 0
    {
        return Err(InferError::Dimension(format!(
            "coop tuilé exige u4, M%{MOE_COOP_TILED_BM}, N%64 et K%64 : u{bits} M={max_m} N={n} K={k}"
        )));
    }
    Ok(())
}

impl MetalExecutor {
    /// GEMM quantifié groupé : `out[M_pad,N] = a[M_pad,K] · deq(W_e[N,K])^T`, l'expert e
    /// de chaque tuile M (16 lignes) lu dans `tile_expert`. Poids empilé `[experts,N,K]`.
    #[expect(clippy::too_many_arguments, reason = "GEMM groupé : buffers + dims")]
    #[allow(
        dead_code,
        reason = "chemin non-fusé, superseded par la variante gather/scatter fusée"
    )]
    fn encode_coop_qb_grouped(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a_bf16: &BufferRef,
        weight: &StackedAffineBuffers,
        out: &BufferRef,
        tile_expert: &BufferRef,
        m_padded: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if weight.bits != 8 || weight.group_size != 64 {
            return Err(InferError::Config(format!(
                "coop grouped exige u8 gs64, reçu bits={} gs={}",
                weight.bits, weight.group_size
            )));
        }
        let pso = self
            .na_gemm_coop_qb_grouped
            .as_ref()
            .ok_or_else(|| InferError::Config("coop_qb_grouped: NA indisponible".into()))?;
        let mnk = [
            checked_u32(m_padded, "coop g m")?,
            checked_u32(n, "coop g n")?,
            checked_u32(k, "coop g k")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(a_bf16), 0);
        encoder.set_buffer(1, Some(&weight.packed), weight.packed_offset);
        encoder.set_buffer(2, Some(&weight.scales), weight.scales_offset);
        encoder.set_buffer(3, Some(&weight.biases), weight.biases_offset);
        encoder.set_buffer(4, Some(out), 0);
        encoder.set_buffer(5, Some(tile_expert), 0);
        encoder.set_bytes(6, 12, mnk.as_ptr().cast::<c_void>());
        trace_dispatch_path("gemm_nax_coop_qb_grouped", m_padded, n, k);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new((m_padded / 16) as u64, n.div_ceil(32) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// GEMM groupé + GATHER fusé (gate/up) : lit `input[token]` f32 via `perm`, zéro a_padded.
    #[expect(
        clippy::too_many_arguments,
        reason = "GEMM groupé fusé : buffers + dims"
    )]
    fn encode_coop_qb_grouped_gather(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        weight: &StackedAffineBuffers,
        out: &BufferRef,
        tile_expert: &BufferRef,
        perm: &BufferRef,
        max_m: usize,
        n: usize,
        k: usize,
        top_k: usize,
    ) -> Result<()> {
        if weight.bits != 8 || weight.group_size != 64 {
            return Err(InferError::Config("coop gather exige u8 gs64".into()));
        }
        let pso = self
            .na_gemm_coop_qb_grouped_gather
            .as_ref()
            .ok_or_else(|| InferError::Config("coop_qb_grouped_gather: NA indispo".into()))?;
        let mnkt = [
            checked_u32(max_m, "g m")?,
            checked_u32(n, "g n")?,
            checked_u32(k, "g k")?,
            checked_u32(top_k, "g tk")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&weight.packed), weight.packed_offset);
        encoder.set_buffer(2, Some(&weight.scales), weight.scales_offset);
        encoder.set_buffer(3, Some(&weight.biases), weight.biases_offset);
        encoder.set_buffer(4, Some(out), 0);
        encoder.set_buffer(5, Some(tile_expert), 0);
        encoder.set_buffer(6, Some(perm), 0);
        encoder.set_bytes(7, 16, mnkt.as_ptr().cast::<c_void>());
        trace_dispatch_path("gemm_nax_coop_qb_grouped_gather", max_m, n, k);
        profile_dispatch();
        encoder.dispatch_thread_groups(
            MTLSize::new((max_m / 16) as u64, n.div_ceil(32) as u64, 1),
            MTLSize::new(32, 1, 1),
        );
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// GEMM groupé gate+up + SwiGLU fusé : écrit directement hidden bf16.
    /// `tiled` choisit le kernel u4 tuilé ([`MOE_COOP_TILED_BM`] × 64), dont
    /// les tuiles de `tile_expert` font [`MOE_COOP_TILED_BM`] lignes.
    #[expect(
        clippy::too_many_arguments,
        reason = "GEMM groupé fusé : buffers + dims"
    )]
    fn encode_coop_qb_grouped_gate_up_swiglu(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        gate: &StackedAffineBuffers,
        up: &StackedAffineBuffers,
        hidden_bf16: &BufferRef,
        tile_expert: &BufferRef,
        perm: &BufferRef,
        max_m: usize,
        n: usize,
        k: usize,
        top_k: usize,
        tiled: bool,
    ) -> Result<()> {
        if gate.bits != up.bits || gate.group_size != 64 || up.group_size != 64 {
            return Err(InferError::Config(format!(
                "coop swiglu fusé exige bits identiques gs64, reçu gate=u{} gs{} up=u{} gs{}",
                gate.bits, gate.group_size, up.bits, up.group_size
            )));
        }
        if gate.experts != up.experts
            || gate.out_dim != up.out_dim
            || gate.in_dim != up.in_dim
            || gate.packed_cols != up.packed_cols
            || gate.groups != up.groups
            || gate.out_dim != n
            || gate.in_dim != k
        {
            return Err(InferError::Dimension(format!(
                "coop swiglu fusé dims gate=[e{} n{} k{}] up=[e{} n{} k{}] attendu n={n} k={k}",
                gate.experts, gate.out_dim, gate.in_dim, up.experts, up.out_dim, up.in_dim
            )));
        }
        let (pso, kernel_name) = if tiled {
            check_coop_tiled_shape(gate.bits, max_m, n, k)?;
            (
                self.na_gemm_coop_qb_grouped_gate_up_swiglu_u4_tiled
                    .as_ref(),
                "gemm_nax_coop_qb_grouped_gate_up_swiglu_u4_tiled",
            )
        } else if gate.bits == FAST_QMV_BITS {
            (
                self.na_gemm_coop_qb_grouped_gate_up_swiglu_u4.as_ref(),
                "gemm_nax_coop_qb_grouped_gate_up_swiglu_u4",
            )
        } else if gate.bits == 8 {
            (
                self.na_gemm_coop_qb_grouped_gate_up_swiglu.as_ref(),
                "gemm_nax_coop_qb_grouped_gate_up_swiglu",
            )
        } else {
            return Err(InferError::Config(format!(
                "coop swiglu fusé bits non supportés u{}",
                gate.bits
            )));
        };
        let pso = pso.ok_or_else(|| InferError::Config(format!("{kernel_name}: NA indispo")))?;
        let mnkt = [
            checked_u32(max_m, "gu m")?,
            checked_u32(n, "gu n")?,
            checked_u32(k, "gu k")?,
            checked_u32(top_k, "gu tk")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(&gate.packed), gate.packed_offset);
        encoder.set_buffer(2, Some(&gate.scales), gate.scales_offset);
        encoder.set_buffer(3, Some(&gate.biases), gate.biases_offset);
        encoder.set_buffer(4, Some(&up.packed), up.packed_offset);
        encoder.set_buffer(5, Some(&up.scales), up.scales_offset);
        encoder.set_buffer(6, Some(&up.biases), up.biases_offset);
        encoder.set_buffer(7, Some(hidden_bf16), 0);
        encoder.set_buffer(8, Some(tile_expert), 0);
        encoder.set_buffer(9, Some(perm), 0);
        encoder.set_bytes(10, 16, mnkt.as_ptr().cast::<c_void>());
        trace_dispatch_path(kernel_name, max_m, n, k);
        profile_dispatch();
        encoder.dispatch_thread_groups(coop_grid(max_m, n, tiled), coop_threads(tiled));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    /// GEMM groupé + SCATTER fusé (down) : écrit `C[slot]` via `perm`, zéro down_res/scatter.
    #[expect(
        clippy::too_many_arguments,
        reason = "GEMM groupé fusé : buffers + dims"
    )]
    fn encode_coop_qb_grouped_scatter(
        &self,
        encoder: &ComputeCommandEncoderRef,
        a_bf16: &BufferRef,
        weight: &StackedAffineBuffers,
        scratch_down: &BufferRef,
        tile_expert: &BufferRef,
        perm: &BufferRef,
        max_m: usize,
        n: usize,
        k: usize,
        tiled: bool,
    ) -> Result<()> {
        if weight.group_size != 64 {
            return Err(InferError::Config(format!(
                "coop scatter exige gs64, reçu bits={} gs={}",
                weight.bits, weight.group_size
            )));
        }
        let (pso, kernel_name) = if tiled {
            check_coop_tiled_shape(weight.bits, max_m, n, k)?;
            (
                self.na_gemm_coop_qb_grouped_scatter_u4_tiled.as_ref(),
                "gemm_nax_coop_qb_grouped_scatter_u4_tiled",
            )
        } else if weight.bits == FAST_QMV_BITS {
            (
                self.na_gemm_coop_qb_grouped_scatter_u4.as_ref(),
                "gemm_nax_coop_qb_grouped_scatter_u4",
            )
        } else if weight.bits == 8 {
            (
                self.na_gemm_coop_qb_grouped_scatter.as_ref(),
                "gemm_nax_coop_qb_grouped_scatter",
            )
        } else {
            return Err(InferError::Config(format!(
                "coop scatter bits non supportés u{}",
                weight.bits
            )));
        };
        let pso = pso.ok_or_else(|| InferError::Config(format!("{kernel_name}: NA indispo")))?;
        let mnk = [
            checked_u32(max_m, "s m")?,
            checked_u32(n, "s n")?,
            checked_u32(k, "s k")?,
        ];
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(a_bf16), 0);
        encoder.set_buffer(1, Some(&weight.packed), weight.packed_offset);
        encoder.set_buffer(2, Some(&weight.scales), weight.scales_offset);
        encoder.set_buffer(3, Some(&weight.biases), weight.biases_offset);
        encoder.set_buffer(4, Some(scratch_down), 0);
        encoder.set_buffer(5, Some(tile_expert), 0);
        encoder.set_buffer(6, Some(perm), 0);
        encoder.set_bytes(7, 12, mnk.as_ptr().cast::<c_void>());
        trace_dispatch_path(kernel_name, max_m, n, k);
        profile_dispatch();
        encoder.dispatch_thread_groups(coop_grid(max_m, n, tiled), coop_threads(tiled));
        post_dispatch_barrier(encoder);
        Ok(())
    }

    #[allow(
        dead_code,
        reason = "gather non-fusé, superseded par le gather fusé en kernel"
    )]
    #[allow(
        clippy::too_many_arguments,
        reason = "encodeur Metal chaud: buffers et dimensions correspondent directement au kernel gather"
    )]
    fn encode_moe_coop_gather(
        &self,
        encoder: &ComputeCommandEncoderRef,
        input: &BufferRef,
        perm: &BufferRef,
        out_bf16: &BufferRef,
        m_padded: usize,
        hidden: usize,
        top_k: usize,
    ) -> Result<()> {
        let pso = self
            .moe_coop_gather_padded
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_coop_gather: indisponible".into()))?;
        let dims = [
            checked_u32(m_padded, "gather mpad")?,
            checked_u32(hidden, "gather hidden")?,
            checked_u32(top_k, "gather top_k")?,
            0,
        ];
        let n = checked_len(m_padded, hidden, "gather n")?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(input), 0);
        encoder.set_buffer(1, Some(perm), 0);
        encoder.set_buffer(2, Some(out_bf16), 0);
        set_u32_bytes(encoder, 3, &dims, "moe_coop_gather_dims")?;
        trace_dispatch_path("moe_coop_gather_padded", m_padded, hidden, top_k);
        self.dispatch_1d(encoder, pso, n)
    }

    #[allow(
        dead_code,
        reason = "scatter non-fusé, superseded par le scatter fusé en kernel"
    )]
    fn encode_moe_coop_scatter(
        &self,
        encoder: &ComputeCommandEncoderRef,
        result: &BufferRef,
        perm: &BufferRef,
        scratch: &BufferRef,
        m_padded: usize,
        out_dim: usize,
    ) -> Result<()> {
        let pso = self
            .moe_coop_scatter_padded
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_coop_scatter: indisponible".into()))?;
        let dims = [
            checked_u32(m_padded, "scatter mpad")?,
            checked_u32(out_dim, "scatter out_dim")?,
            0,
            0,
        ];
        let n = checked_len(m_padded, out_dim, "scatter n")?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(result), 0);
        encoder.set_buffer(1, Some(perm), 0);
        encoder.set_buffer(2, Some(scratch), 0);
        set_u32_bytes(encoder, 3, &dims, "moe_coop_scatter_dims")?;
        trace_dispatch_path("moe_coop_scatter_padded", m_padded, out_dim, 0);
        self.dispatch_1d(encoder, pso, n)
    }

    fn encode_moe_g_fill(
        &self,
        encoder: &ComputeCommandEncoderRef,
        buf: &BufferRef,
        n: usize,
        value: u32,
    ) -> Result<()> {
        let pso = self
            .moe_g_fill_u32
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_g_fill: indisponible".into()))?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(buf), 0);
        set_u32_bytes(
            encoder,
            1,
            &[checked_u32(n, "fill n")?, value],
            "moe_g_fill_nv",
        )?;
        trace_dispatch_path("moe_g_fill_u32", n, 1, 0);
        self.dispatch_1d(encoder, pso, n)
    }

    fn encode_moe_g_histogram(
        &self,
        encoder: &ComputeCommandEncoderRef,
        indices: &BufferRef,
        counts: &BufferRef,
        total: usize,
    ) -> Result<()> {
        let pso = self
            .moe_g_histogram
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_g_histogram: indisponible".into()))?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(indices), 0);
        encoder.set_buffer(1, Some(counts), 0);
        set_u32_bytes(
            encoder,
            2,
            &[checked_u32(total, "hist total")?],
            "moe_g_hist_total",
        )?;
        trace_dispatch_path("moe_g_histogram", total, 1, 0);
        self.dispatch_1d(encoder, pso, total)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "kernel de grouping : buffers + dimensions"
    )]
    fn encode_moe_g_offsets(
        &self,
        encoder: &ComputeCommandEncoderRef,
        counts: &BufferRef,
        padded_offset: &BufferRef,
        tile_expert: &BufferRef,
        cursor: &BufferRef,
        experts: usize,
        tile_m: usize,
    ) -> Result<()> {
        let pso = self
            .moe_g_offsets
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_g_offsets: indisponible".into()))?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(counts), 0);
        encoder.set_buffer(1, Some(padded_offset), 0);
        encoder.set_buffer(2, Some(tile_expert), 0);
        encoder.set_buffer(3, Some(cursor), 0);
        set_u32_bytes(
            encoder,
            4,
            &[
                checked_u32(experts, "offsets experts")?,
                checked_u32(tile_m, "offsets tile_m")?,
            ],
            "moe_g_off_e",
        )?;
        trace_dispatch_path("moe_g_offsets", 1, experts, 0);
        self.dispatch_1d(encoder, pso, 1)
    }

    fn encode_moe_g_perm(
        &self,
        encoder: &ComputeCommandEncoderRef,
        indices: &BufferRef,
        cursor: &BufferRef,
        perm: &BufferRef,
        total: usize,
    ) -> Result<()> {
        let pso = self
            .moe_g_perm
            .as_ref()
            .ok_or_else(|| InferError::Config("moe_g_perm: indisponible".into()))?;
        encoder.set_compute_pipeline_state(pso);
        encoder.set_buffer(0, Some(indices), 0);
        encoder.set_buffer(1, Some(cursor), 0);
        encoder.set_buffer(2, Some(perm), 0);
        set_u32_bytes(
            encoder,
            3,
            &[checked_u32(total, "perm total")?],
            "moe_g_perm_total",
        )?;
        trace_dispatch_path("moe_g_perm", total, 1, 0);
        self.dispatch_1d(encoder, pso, total)
    }

    /// Renvoie vrai si tous les pipelines du chemin routed coop existent pour
    /// cette quantification. Les kernels groupés exigent MPP (macOS 26) : sans
    /// eux, l'appelant garde le chemin gather qmv au lieu d'échouer.
    pub(crate) fn routed_coop_pipelines_ready(&self, stacked: &StackedMoeBuffers) -> bool {
        let u4 = stacked.gate.bits == FAST_QMV_BITS;
        let gate_up = if moe_coop_fused_swiglu_enabled() {
            if u4 {
                &self.na_gemm_coop_qb_grouped_gate_up_swiglu_u4
            } else {
                &self.na_gemm_coop_qb_grouped_gate_up_swiglu
            }
        } else {
            // Le gather non fusionné n'existe qu'en u8.
            &self.na_gemm_coop_qb_grouped_gather
        };
        let scatter = if u4 {
            &self.na_gemm_coop_qb_grouped_scatter_u4
        } else {
            &self.na_gemm_coop_qb_grouped_scatter
        };
        [
            gate_up,
            scatter,
            &self.moe_g_fill_u32,
            &self.moe_g_histogram,
            &self.moe_g_offsets,
            &self.moe_g_perm,
        ]
        .iter()
        .all(|pso| pso.is_some())
    }

    /// Vrai si les kernels tuilés servent ces poids : u4 gs64, gate+up
    /// fusionné, pipelines compilés et dimensions multiples de 64 (tuile N et
    /// pas de K des kernels, sans masque).
    fn routed_coop_tiled_ready(&self, stacked: &StackedMoeBuffers) -> bool {
        let aligned = |b: &StackedAffineBuffers| {
            b.out_dim % COOP_TILED_BN == 0 && b.in_dim % COOP_TILED_BN == 0
        };
        stacked.gate.bits == FAST_QMV_BITS
            && moe_coop_fused_swiglu_enabled()
            && aligned(&stacked.gate)
            && aligned(&stacked.up)
            && aligned(&stacked.down)
            && self
                .na_gemm_coop_qb_grouped_gate_up_swiglu_u4_tiled
                .is_some()
            && self.na_gemm_coop_qb_grouped_scatter_u4_tiled.is_some()
    }

    /// Vrai si la bibliothèque NA (MPP, macOS 26) compile sur cette machine :
    /// les pipelines coop groupés en font partie et doivent alors exister.
    #[cfg(test)]
    pub(crate) fn na_library_available_for_test(&self) -> bool {
        self.na_gemm_bf16.is_some()
    }

    /// Retire le pipeline scatter u4 du coop groupé pour exercer le repli qmv.
    #[cfg(test)]
    pub(crate) fn drop_routed_coop_scatter_u4_for_test(&mut self) {
        self.na_gemm_coop_qb_grouped_scatter_u4 = None;
    }

    /// Encode MoE routed-only prefill, experts routés sur tensor-cores, dans
    /// l'encoder appelant. Réutilise les buffers rows déjà alloués par le résident.
    ///
    /// `router_score` choisit le top-k (softmax Qwen ou sigmoïde renormalisée
    /// flint). Les kernels groupés ne lisent jamais les scores : seule la somme
    /// pondérée finale les consomme, dans le layout `[rows, top_k]` d'origine.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si shapes divergent, NA indispo, quantification non
    /// supportée, ou Metal échoue.
    #[expect(
        clippy::too_many_arguments,
        reason = "chemin MoE routed rows coop : buffers scratch + poids"
    )]
    pub(crate) fn encode_moe_routed_rows_coop(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        router_buffer: &BufferRef,
        indices_buffer: &BufferRef,
        scores_buffer: &BufferRef,
        down_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeRoutedWeights,
        top_k: usize,
        router_score: crate::RouterScore,
    ) -> Result<()> {
        let shape = self.check_moe_routed_buffer_shapes(in_dim, weights, top_k)?;
        let experts = shape.expert_count;
        let inter = shape.inter_dim;
        let out_dim = shape.out_dim;
        trace_dispatch_path("moe_routed_rows_coop", rows, out_dim, in_dim);
        let total = checked_len(rows, top_k, "moe routed coop total")?;

        // Kernels tuilés : flint (sigmoïde) seulement, cf. `flint_moe_coop_tiled_enabled`.
        let tiled = matches!(router_score, crate::RouterScore::Sigmoid)
            && flint_moe_coop_tiled_enabled()
            && self.routed_coop_tiled_ready(&weights.stacked);
        #[cfg(test)]
        if tiled {
            MOE_COOP_TILED_TRACE.with(|trace| trace.set(trace.get() + 1));
        }
        let tile_m = if tiled {
            MOE_COOP_TILED_BM
        } else {
            MOE_COOP_TILE_M
        };
        // Pire cas sans readback : chaque expert padé au multiple de `tile_m`
        // au-dessus, soit au plus une tuile partielle par expert.
        let max_tiles = total.div_ceil(tile_m) + experts;
        let max_m = checked_len(max_tiles, tile_m, "moe routed coop max_m")?;
        const SENTINEL: u32 = 0xFFFF_FFFF;
        let counts = self.private_u32_buffer(experts, "moe_routed_coop_counts")?;
        let padded_offset = self.private_u32_buffer(experts, "moe_routed_coop_padded_offset")?;
        let cursor = self.private_u32_buffer(experts, "moe_routed_coop_cursor")?;
        let tile_expert = self.private_u32_buffer(max_tiles, "moe_routed_coop_tile_expert")?;
        let perm = self.private_u32_buffer(max_m, "moe_routed_coop_perm")?;
        let hidden2_bf16 = self.private_bf16_buffer(
            checked_len(max_m, inter, "moe routed coop h2b")?,
            "moe_routed_coop_h2b",
        )?;

        let router_out = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.router,
            router_buffer,
            false,
        )?;
        if router_out != experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE routed coop sort {router_out}, attendu {experts}"
            )));
        }
        match router_score {
            crate::RouterScore::Softmax => self.encode_topk_softmax_rows(
                encoder,
                router_buffer,
                indices_buffer,
                scores_buffer,
                rows,
                experts,
                top_k,
            )?,
            crate::RouterScore::Sigmoid => self.encode_topk_sigmoid_rows(
                encoder,
                router_buffer,
                indices_buffer,
                scores_buffer,
                rows,
                experts,
                top_k,
            )?,
        }
        self.encode_moe_g_fill(encoder, &counts, experts, 0)?;
        self.encode_moe_g_fill(encoder, &tile_expert, max_tiles, SENTINEL)?;
        self.encode_moe_g_fill(encoder, &perm, max_m, SENTINEL)?;
        self.encode_moe_g_histogram(encoder, indices_buffer, &counts, total)?;
        self.encode_moe_g_offsets(
            encoder,
            &counts,
            &padded_offset,
            &tile_expert,
            &cursor,
            experts,
            tile_m,
        )?;
        self.encode_moe_g_perm(encoder, indices_buffer, &cursor, &perm, total)?;

        if moe_coop_fused_swiglu_enabled() {
            self.encode_coop_qb_grouped_gate_up_swiglu(
                encoder,
                input_buffer,
                &weights.stacked.gate,
                &weights.stacked.up,
                &hidden2_bf16,
                &tile_expert,
                &perm,
                max_m,
                inter,
                in_dim,
                top_k,
                tiled,
            )?;
        } else {
            let gate_out = self.private_f32_buffer(
                checked_len(max_m, inter, "moe routed coop gate")?,
                "moe_routed_coop_gate",
            )?;
            let up_out = self.private_f32_buffer(
                checked_len(max_m, inter, "moe routed coop up")?,
                "moe_routed_coop_up",
            )?;
            let hidden2 = self.private_f32_buffer(
                checked_len(max_m, inter, "moe routed coop h2")?,
                "moe_routed_coop_h2",
            )?;
            self.encode_coop_qb_grouped_gather(
                encoder,
                input_buffer,
                &weights.stacked.gate,
                &gate_out,
                &tile_expert,
                &perm,
                max_m,
                inter,
                in_dim,
                top_k,
            )?;
            self.encode_coop_qb_grouped_gather(
                encoder,
                input_buffer,
                &weights.stacked.up,
                &up_out,
                &tile_expert,
                &perm,
                max_m,
                inter,
                in_dim,
                top_k,
            )?;
            let swn = checked_len(max_m, inter, "moe routed coop swiglu")?;
            encoder.set_compute_pipeline_state(&self.swiglu_f32);
            encoder.set_buffer(0, Some(&gate_out), 0);
            encoder.set_buffer(1, Some(&up_out), 0);
            encoder.set_buffer(2, Some(&hidden2), 0);
            set_u32_bytes(
                encoder,
                3,
                &[checked_u32(swn, "swn")?],
                "moe_routed_coop_swn",
            )?;
            trace_dispatch_path("swiglu_f32", swn, 1, 0);
            self.dispatch_1d(encoder, &self.swiglu_f32, swn)?;
            self.encode_f32_to_bf16(encoder, &hidden2, &hidden2_bf16, swn)?;
        }
        self.encode_coop_qb_grouped_scatter(
            encoder,
            &hidden2_bf16,
            &weights.stacked.down,
            down_buffer,
            &tile_expert,
            &perm,
            max_m,
            out_dim,
            inter,
            tiled,
        )?;
        match residual {
            Some(r) => self.encode_weighted_sum_add_grouped_topk(
                encoder,
                owned_buffers,
                down_buffer,
                scores_buffer,
                r,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
            None => self.encode_weighted_sum_grouped_topk(
                encoder,
                owned_buffers,
                down_buffer,
                scores_buffer,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
        }
        Ok(())
    }

    /// Encode MoE shared-expert prefill, experts routés sur tensor-cores (port
    /// `gather_qmm`), dans l'encoder appelant.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si shapes divergent, NA indispo, quant ≠ u8 gs64, ou Metal échoue.
    #[allow(
        clippy::too_many_arguments,
        reason = "encodeur MoE chaud: buffers, dimensions et poids restent alignés sur les bindings Metal"
    )]
    pub(crate) fn encode_moe_shared_rows_coop(
        &self,
        encoder: &ComputeCommandEncoderRef,
        owned_buffers: &mut Vec<Buffer>,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<()> {
        let shape = self.check_moe_shared_buffer_shapes(in_dim, weights, top_k)?;
        let experts = shape.expert_count;
        let inter = shape.inter_dim;
        let out_dim = shape.out_dim;
        let shared_inter = shape.shared_inter_dim;
        let hidden = in_dim;
        trace_dispatch_path("moe_shared_rows_coop", rows, out_dim, in_dim);
        let total = checked_len(rows, top_k, "moe coop total")?;
        let scratch = self.allocate_moe_shared_rows_scratch(
            rows,
            top_k,
            experts,
            inter,
            out_dim,
            shared_inter,
        )?;

        // Dimensionnement WORST-CASE (zéro readback) : chaque expert padé à 16 lignes.
        let max_tiles = total.div_ceil(16) + experts;
        let max_m = checked_len(max_tiles, 16, "moe coop max_m")?;
        const SENTINEL: u32 = 0xFFFF_FFFF;
        // Buffers de grouping GPU.
        let counts = self.private_u32_buffer(experts, "moe_coop_counts")?;
        let padded_offset = self.private_u32_buffer(experts, "moe_coop_padded_offset")?;
        let cursor = self.private_u32_buffer(experts, "moe_coop_cursor")?;
        let tile_expert = self.private_u32_buffer(max_tiles, "moe_coop_tile_expert")?;
        let perm = self.private_u32_buffer(max_m, "moe_coop_perm")?;
        // Scratch padé (taille worst-case). gather/scatter FUSÉS → ni a_padded ni down_res.
        let gate_out =
            self.private_f32_buffer(checked_len(max_m, inter, "moe coop go")?, "moe_coop_go")?;
        let up_out =
            self.private_f32_buffer(checked_len(max_m, inter, "moe coop uo")?, "moe_coop_uo")?;
        let hidden2 =
            self.private_f32_buffer(checked_len(max_m, inter, "moe coop h2")?, "moe_coop_h2")?;
        let hidden2_bf16 =
            self.private_bf16_buffer(checked_len(max_m, inter, "moe coop h2b")?, "moe_coop_h2b")?;

        // Routeur + top-k.
        let router_out = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.router,
            &scratch.router,
            false,
        )?;
        if router_out != experts {
            return Err(InferError::Dimension(format!(
                "routeur MoE coop sort {router_out}, attendu {experts}"
            )));
        }
        self.encode_topk_softmax_rows(
            encoder,
            &scratch.router,
            &scratch.indices,
            &scratch.scores,
            rows,
            experts,
            top_k,
        )?;
        // Grouping GPU : zero/sentinel → histogram → scan offsets → build perm.
        self.encode_moe_g_fill(encoder, &counts, experts, 0)?;
        self.encode_moe_g_fill(encoder, &tile_expert, max_tiles, SENTINEL)?;
        self.encode_moe_g_fill(encoder, &perm, max_m, SENTINEL)?;
        self.encode_moe_g_histogram(encoder, &scratch.indices, &counts, total)?;
        self.encode_moe_g_offsets(
            encoder,
            &counts,
            &padded_offset,
            &tile_expert,
            &cursor,
            experts,
            MOE_COOP_TILE_M,
        )?;
        self.encode_moe_g_perm(encoder, &scratch.indices, &cursor, &perm, total)?;
        // gate/up : GEMM + GATHER fusé (lit input via perm). swiglu padé.
        // down : GEMM + SCATTER fusé (écrit scratch.down[slot] via perm).
        // DIAGNOSTIC perf : SARAGOSSA_RUST_MOE_COOP_SKIP_GEMM saute les 3 grouped GEMMs
        // (sortie FAUSSE) → le delta vs la version complète = le coût du GEMM routé.
        let skip_gemm =
            crate::runtime_flags::env_var("SARAGOSSA_RUST_MOE_COOP_SKIP_GEMM").is_some();
        let fused_swiglu = !skip_gemm && moe_coop_fused_swiglu_enabled();
        if skip_gemm {
            // Zéro gate/up (0.0f32 = 0u32) pour éviter les dénormaux qui faussent le timing.
            self.encode_moe_g_fill(encoder, &gate_out, checked_len(max_m, inter, "skip go")?, 0)?;
            self.encode_moe_g_fill(encoder, &up_out, checked_len(max_m, inter, "skip uo")?, 0)?;
        } else if fused_swiglu {
            self.encode_coop_qb_grouped_gate_up_swiglu(
                encoder,
                input_buffer,
                &weights.stacked.gate,
                &weights.stacked.up,
                &hidden2_bf16,
                &tile_expert,
                &perm,
                max_m,
                inter,
                hidden,
                top_k,
                false,
            )?;
        } else {
            self.encode_coop_qb_grouped_gather(
                encoder,
                input_buffer,
                &weights.stacked.gate,
                &gate_out,
                &tile_expert,
                &perm,
                max_m,
                inter,
                hidden,
                top_k,
            )?;
            self.encode_coop_qb_grouped_gather(
                encoder,
                input_buffer,
                &weights.stacked.up,
                &up_out,
                &tile_expert,
                &perm,
                max_m,
                inter,
                hidden,
                top_k,
            )?;
        }
        let swn = checked_len(max_m, inter, "moe coop swiglu")?;
        if !fused_swiglu {
            encoder.set_compute_pipeline_state(&self.swiglu_f32);
            encoder.set_buffer(0, Some(&gate_out), 0);
            encoder.set_buffer(1, Some(&up_out), 0);
            encoder.set_buffer(2, Some(&hidden2), 0);
            set_u32_bytes(encoder, 3, &[checked_u32(swn, "swn")?], "moe_coop_swn")?;
            trace_dispatch_path("swiglu_f32", swn, 1, 0);
            self.dispatch_1d(encoder, &self.swiglu_f32, swn)?;
            self.encode_f32_to_bf16(encoder, &hidden2, &hidden2_bf16, swn)?;
        }
        if !skip_gemm {
            self.encode_coop_qb_grouped_scatter(
                encoder,
                &hidden2_bf16,
                &weights.stacked.down,
                &scratch.down,
                &tile_expert,
                &perm,
                max_m,
                hidden,
                inter,
                false,
            )?;
        }
        // Expert partagé + combine (même encodeur). DIAGNOSTIC : SKIP_SHARED saute
        // les 3 projections shared (par-token, qmv) → le delta = leur coût.
        let skip_shared =
            crate::runtime_flags::env_var("SARAGOSSA_RUST_MOE_COOP_SKIP_SHARED").is_some();
        let gate_dim = self.encode_matmul_weight_buffers(
            encoder,
            input_buffer,
            rows,
            in_dim,
            &weights.shared_gate,
            &scratch.shared_gate,
            false,
        )?;
        if gate_dim != 1 {
            return Err(InferError::Dimension(format!(
                "shared gate coop sort {gate_dim}"
            )));
        }
        if !skip_shared {
            let sg = self.encode_matmul_weight_buffers(
                encoder,
                input_buffer,
                rows,
                in_dim,
                &weights.shared_gate_proj,
                &scratch.shared_proj_gate,
                false,
            )?;
            let su = self.encode_matmul_weight_buffers(
                encoder,
                input_buffer,
                rows,
                in_dim,
                &weights.shared_up_proj,
                &scratch.shared_up,
                false,
            )?;
            if sg != shared_inter || su != shared_inter {
                return Err(InferError::Dimension(format!(
                    "shared proj coop gate={sg} up={su}"
                )));
            }
            self.encode_swiglu(
                encoder,
                owned_buffers,
                &scratch.shared_proj_gate,
                &scratch.shared_up,
                &scratch.shared_hidden,
                checked_len(rows, shared_inter, "moe coop shared swiglu")?,
            )?;
            let sd = self.encode_matmul_weight_buffers(
                encoder,
                &scratch.shared_hidden,
                rows,
                shared_inter,
                &weights.shared_down_proj,
                &scratch.shared_down,
                false,
            )?;
            if sd != out_dim {
                return Err(InferError::Dimension(format!("shared down coop sort {sd}")));
            }
        }
        match residual {
            Some(r) => self.encode_weighted_sum_add_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                r,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
            None => self.encode_weighted_sum_grouped_topk(
                encoder,
                owned_buffers,
                &scratch.down,
                &scratch.scores,
                output_buffer,
                rows,
                top_k,
                out_dim,
            )?,
        }
        if !skip_shared {
            self.encode_add_sigmoid_scaled_rows(
                encoder,
                &scratch.shared_down,
                &scratch.shared_gate,
                output_buffer,
                rows,
                out_dim,
            )?;
        }
        Ok(())
    }

    /// MoE shared-expert prefill, experts routés sur tensor-cores (port `gather_qmm`).
    /// `input`/`residual` COMMITÉS. Crée+commit son propre command buffer.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si shapes divergent, NA indispo, quant ≠ u8 gs64, ou Metal échoue.
    #[allow(
        clippy::too_many_arguments,
        reason = "wrapper MoE chaud: regrouper buffers et dimensions ajouterait un objet intermédiaire au prefill"
    )]
    pub(crate) fn moe_shared_rows_coop(
        &self,
        input_buffer: &BufferRef,
        residual: Option<&BufferRef>,
        output_buffer: &BufferRef,
        rows: usize,
        in_dim: usize,
        weights: &MetalMoeSharedWeights,
        top_k: usize,
    ) -> Result<()> {
        let cb = self.queue.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        let guard = EncoderEndGuard::new(enc);
        let mut owned: Vec<Buffer> = Vec::new();
        self.encode_moe_shared_rows_coop(
            enc,
            &mut owned,
            input_buffer,
            residual,
            output_buffer,
            rows,
            in_dim,
            weights,
            top_k,
        )?;
        guard.end();
        set_commit_label("moe_coop");
        commit_and_wait(cb)
    }
}
