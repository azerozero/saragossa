//! Types de poids et de profilage du decode résident.

use super::*;

/// Poids d'UNE couche full-attn MoE (decode résident 1c) : handles Metal résolus
/// au setup de génération, sans lookup de cache dans le chemin par token.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub qkv_proj: Option<&'a MetalLinearWeightBuffers>,
    pub q_proj: &'a MetalLinearWeightBuffers,
    pub k_proj: &'a MetalLinearWeightBuffers,
    pub v_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub pre_feedforward_norm: Option<&'a Buffer>,
    pub post_feedforward_norm: Option<&'a Buffer>,
    pub layer_scalar: Option<f32>,
    pub moe: &'a MetalMoeSharedWeights,
    pub top_k: usize,
}

/// Poids d'UNE couche full-attn MoE routed-only résidente.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnRoutedLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    /// Le chemin routed-only résident est construit uniquement quand le QKV
    /// concaténé est disponible. Sinon le setup refuse le decode résident avant
    /// d'instancier ce type, ce qui rend le fallback 3-matmuls inatteignable ici.
    pub qkv_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub pre_feedforward_norm: Option<&'a Buffer>,
    pub post_feedforward_norm: Option<&'a Buffer>,
    pub layer_scalar: Option<f32>,
    pub moe: &'a MetalMoeRoutedWeights,
    pub top_k: usize,
}

/// Poids d'UNE couche full-attn dense (decode résident 1c). Structure séparée
/// du chemin MoE pour garder les préconditions et les poids MoE inchangés.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnDenseLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub qkv_proj: Option<&'a MetalLinearWeightBuffers>,
    /// Force le buffer concaténé pour le layout K=V sans `v_proj` séparé.
    pub qkv_proj_without_gate: bool,
    pub value_norm: bool,
    pub q_proj: &'a MetalLinearWeightBuffers,
    pub k_proj: &'a MetalLinearWeightBuffers,
    pub v_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub pre_feedforward_norm: Option<&'a Buffer>,
    pub post_feedforward_norm: Option<&'a Buffer>,
    pub layer_scalar: Option<f32>,
    pub gate_proj: &'a MetalLinearWeightBuffers,
    pub up_proj: &'a MetalLinearWeightBuffers,
    pub down_proj: &'a MetalLinearWeightBuffers,
    pub tail_score: &'a Buffer,
    pub parallel_moe: Option<GemmaParallelMoeTailWeights<'a>>,
}

/// Dimensions + paramètres RoPE d'UNE couche full-attn de decode résident.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnLayerDims {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub attn_scalar: f32,
    pub rope_dims: usize,
    pub rope_frequency_dim: usize,
    pub position: usize,
    /// Première ligne KV incluse dans le softmax (`0` = contexte complet).
    pub window_start: usize,
    pub eps: f32,
    pub theta: f32,
    pub attn_output_gate: bool,
}

/// Poids d'UNE couche linear-attn MoE (decode résident 1c) : handles Metal
/// résolus au setup de génération, sans lookup de cache dans le chemin par token.
#[derive(Clone, Copy)]
pub(crate) struct LinearAttnLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub linear: &'a MetalLinearAttnResidentDenseWeights,
    pub post_norm: &'a Buffer,
    pub moe: &'a MetalMoeSharedWeights,
    pub top_k: usize,
}

/// Poids d'UNE couche linear-attn dense (decode résident 1c). Structure séparée
/// du chemin MoE.
#[derive(Clone, Copy)]
pub(crate) struct LinearAttnDenseLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub linear: &'a MetalLinearAttnResidentDenseWeights,
    pub post_norm: &'a Buffer,
    pub gate_proj: &'a MetalLinearWeightBuffers,
    pub up_proj: &'a MetalLinearWeightBuffers,
    pub down_proj: &'a MetalLinearWeightBuffers,
    pub tail_score: &'a Buffer,
}

/// Poids additionnels du tail FFN dense Gemma 4.
#[derive(Clone, Copy)]
pub(crate) struct GemmaDenseTailWeights<'a> {
    pub pre_feedforward_norm: &'a Buffer,
    pub post_feedforward_norm: &'a Buffer,
    pub layer_scalar: Option<f32>,
}

/// Poids additionnels du tail MoE routed Gemma 4.
#[derive(Clone, Copy)]
pub(crate) struct GemmaMoeTailWeights<'a> {
    pub pre_feedforward_norm: &'a Buffer,
    pub post_feedforward_norm: &'a Buffer,
    pub layer_scalar: Option<f32>,
    pub moe: &'a MetalMoeRoutedWeights,
    pub top_k: usize,
}

/// Poids du tail dense + MoE parallèle Gemma 4.
#[derive(Clone, Copy)]
pub(crate) struct GemmaParallelMoeTailWeights<'a> {
    pub dense_gate_proj: &'a MetalLinearWeightBuffers,
    pub dense_up_proj: &'a MetalLinearWeightBuffers,
    pub dense_down_proj: &'a MetalLinearWeightBuffers,
    pub pre_feedforward_norm: &'a Buffer,
    pub post_feedforward_norm_1: &'a Buffer,
    pub moe: &'a MetalMoeRoutedWeights,
    pub top_k: usize,
    pub router_norm: Option<(&'a BufferRef, f32)>,
    pub per_expert_scale: Option<&'a BufferRef>,
    pub pre_feedforward_norm_2: &'a Buffer,
    pub post_feedforward_norm_2: &'a Buffer,
    pub post_feedforward_norm: &'a Buffer,
    pub layer_scalar: Option<f32>,
    pub dense_inter_dim: usize,
}

/// Cumul CPU des temps par section du decode résident (ÉTAPE 0 tranche 3,
/// fallback `SARAGOSSA_RUST_GPU_COUNTERS`). Sur Apple Silicon (AGX) l'échantillonnage
/// programmatique de compteurs GPU (`sampleCountersInBuffer`) est **refusé par le
/// device** → on segmente le forward en command buffers par section, chronométrés
/// CPU (`commit_and_wait` borné). Chaque couche = 1 CB (full-attn ou linear-attn,
/// MoE incluse), lm_head = 1 CB. Le surcoût commit/wait par section est UNIFORME
/// par couche → il s'annule dans la différence full↔linear et le rapport
/// privilégie les **%** ; l'isolation MoE vient d'un microbench séparé.
#[derive(Debug, Default, Clone, Copy)]
struct SectionAccum {
    full_layer_ns: u128,
    full_layer_count: u64,
    linear_layer_ns: u128,
    linear_layer_count: u64,
    lmhead_ns: u128,
    tokens: u64,
}

/// Chronomètre per-section (CPU) du decode résident segmenté.
///
/// N'altère jamais le résultat numérique : la segmentation ne change que le
/// découpage en command buffers (mêmes kernels, même ordre, état GPU persistant
/// inchangé entre CBs).
#[derive(Debug, Default)]
pub(crate) struct GpuSectionTimer {
    accum: RefCell<SectionAccum>,
    mtp_cycles: RefCell<[(u128, u64); 2]>,
    mtp_flush: RefCell<(u128, u64)>,
    mtp_d2: RefCell<MtpD2Segments>,
}

/// Temps CPU submit/wait des segments d'UN cycle D2 fusionné (ns), ou leur somme
/// sur plusieurs cycles : append différé, tête MTP (deux drafts), couches de
/// vérification full et linear, queue (norme finale, lm_head, copies).
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MtpD2Segments {
    pub(crate) append_ns: u128,
    pub(crate) head_ns: u128,
    pub(crate) verify_full_ns: u128,
    pub(crate) verify_full_layers: u64,
    pub(crate) verify_linear_ns: u128,
    pub(crate) verify_linear_layers: u64,
    pub(crate) tail_ns: u128,
    pub(crate) cycles: u64,
}

impl MtpD2Segments {
    fn total_ns(&self) -> u128 {
        self.append_ns + self.head_ns + self.verify_full_ns + self.verify_linear_ns + self.tail_ns
    }
}

impl GpuSectionTimer {
    /// Construit le timer si `SARAGOSSA_RUST_GPU_COUNTERS` est défini, sinon `None`
    /// (le decode reste le chemin résident à command buffer unique, inchangé).
    pub(crate) fn try_new() -> Option<Self> {
        crate::runtime_flags::gpu_counters_enabled().then_some(())?;
        eprintln!("gpu sections: actif (AR et cycle MTP D2 segmentés, chronométrage CPU)");
        Some(Self::default())
    }

    /// Enregistre le temps d'UNE couche (full ou linear, MoE incluse).
    pub(crate) fn record_layer(&self, is_full: bool, elapsed_ns: u128) {
        let mut accum = self.accum.borrow_mut();
        if is_full {
            accum.full_layer_ns += elapsed_ns;
            accum.full_layer_count += 1;
        } else {
            accum.linear_layer_ns += elapsed_ns;
            accum.linear_layer_count += 1;
        }
    }

    /// Enregistre le temps de la section lm_head (final_norm + lm_head + argmax) et
    /// clôt le token.
    pub(crate) fn record_lmhead(&self, elapsed_ns: u128) {
        let mut accum = self.accum.borrow_mut();
        accum.lmhead_ns += elapsed_ns;
        accum.tokens += 1;
    }

    /// Mesure CPU submit/wait du cycle fusionné, sans segmenter ses kernels.
    pub(crate) fn record_mtp_cycle(&self, depth: usize, elapsed_ns: u128) {
        if let Some((total, count)) = self.mtp_cycles.borrow_mut().get_mut(match depth {
            1 => 0,
            2 => 1,
            _ => return,
        }) {
            *total += elapsed_ns;
            *count += 1;
        }
    }

    /// Segments d'un cycle D2 fusionné ; le total alimente aussi le compteur de cycles.
    pub(crate) fn record_mtp_d2_segments(&self, segments: MtpD2Segments) {
        {
            let mut accum = self.mtp_d2.borrow_mut();
            accum.append_ns += segments.append_ns;
            accum.head_ns += segments.head_ns;
            accum.verify_full_ns += segments.verify_full_ns;
            accum.verify_full_layers += segments.verify_full_layers;
            accum.verify_linear_ns += segments.verify_linear_ns;
            accum.verify_linear_layers += segments.verify_linear_layers;
            accum.tail_ns += segments.tail_ns;
            accum.cycles += segments.cycles;
        }
        self.record_mtp_cycle(2, segments.total_ns());
    }

    /// Coût supplémentaire du flush d'historique avant un changement de chemin.
    pub(crate) fn record_mtp_flush(&self, elapsed_ns: u128) {
        let mut accum = self.mtp_flush.borrow_mut();
        accum.0 += elapsed_ns;
        accum.1 += 1;
    }

    /// Formate le classement per-section (ms/token + %), ou `None` si rien
    /// d'accumulé. Donne les moyennes par couche full vs linear et le total lm_head.
    pub(crate) fn report(&self) -> Option<String> {
        let accum = *self.accum.borrow();
        let mut mtp = self
            .mtp_cycles
            .borrow()
            .iter()
            .enumerate()
            .filter(|(_, (_, count))| *count > 0)
            .map(|(index, (total, count))| {
                format!(
                    "MTP fused D{} (n={count}, CPU submit/wait, non segmenté): {:.3} ms/cycle",
                    index + 1,
                    *total as f64 / 1.0e6 / *count as f64
                )
            })
            .collect::<Vec<_>>()
            .join(" | ");
        let (total, count) = *self.mtp_flush.borrow();
        if count > 0 {
            if !mtp.is_empty() {
                mtp.push_str(" | ");
            }
            mtp.push_str(&format!(
                "MTP history flush (n={count}, CPU submit/wait, supplément aux cycles): {:.3} ms/flush",
                total as f64 / 1.0e6 / count as f64
            ));
        }
        let d2 = *self.mtp_d2.borrow();
        if d2.cycles > 0 {
            let cycles = d2.cycles as f64;
            let ms = |ns: u128| ns as f64 / 1.0e6 / cycles;
            let per_layer = |ns: u128, layers: u64| {
                if layers > 0 {
                    ns as f64 / 1.0e6 / layers as f64
                } else {
                    0.0
                }
            };
            mtp.push_str(&format!(
                " | MTP D2 segments/cycle (n={}) : append {:.3} ms | tête {:.3} ms | \
                 verify full-attn {:.3} ms ({:.3} ms/couche ×{}) | \
                 verify linear-attn {:.3} ms ({:.3} ms/couche ×{}) | queue {:.3} ms",
                d2.cycles,
                ms(d2.append_ns),
                ms(d2.head_ns),
                ms(d2.verify_full_ns),
                per_layer(d2.verify_full_ns, d2.verify_full_layers),
                d2.verify_full_layers / d2.cycles,
                ms(d2.verify_linear_ns),
                per_layer(d2.verify_linear_ns, d2.verify_linear_layers),
                d2.verify_linear_layers / d2.cycles,
                ms(d2.tail_ns),
            ));
        }
        if accum.tokens == 0 {
            return (!mtp.is_empty()).then_some(mtp);
        }
        let tokens = accum.tokens as f64;
        let ms = |ns: u128| ns as f64 / 1.0e6 / tokens;
        let full_total = ms(accum.full_layer_ns);
        let linear_total = ms(accum.linear_layer_ns);
        let lmhead = ms(accum.lmhead_ns);
        let sum = full_total + linear_total + lmhead;
        let pct = |part: f64| if sum > 0.0 { part / sum * 100.0 } else { 0.0 };
        let full_per = if accum.full_layer_count > 0 {
            accum.full_layer_ns as f64 / 1.0e6 / accum.full_layer_count as f64
        } else {
            0.0
        };
        let linear_per = if accum.linear_layer_count > 0 {
            accum.linear_layer_ns as f64 / 1.0e6 / accum.linear_layer_count as f64
        } else {
            0.0
        };
        Some(format!(
            "gpu sections/token (n={tok}, segmenté CPU) : \
             couches full-attn {full_total:.3} ms ({full_p:.1}%, {full_per:.3} ms/couche ×{nf}) | \
             couches linear-attn {linear_total:.3} ms ({linear_p:.1}%, {linear_per:.3} ms/couche ×{nl}) | \
             lm_head {lmhead:.3} ms ({lmhead_p:.1}%) | Σ {sum:.3} ms/token{mtp_suffix}",
            mtp_suffix = if mtp.is_empty() {
                String::new()
            } else {
                format!(" | {mtp}")
            },
            tok = accum.tokens,
            nf = accum.full_layer_count / accum.tokens,
            nl = accum.linear_layer_count / accum.tokens,
            full_p = pct(full_total),
            linear_p = pct(linear_total),
            lmhead_p = pct(lmhead),
        ))
    }
}

#[cfg(test)]
mod timer_tests {
    use super::GpuSectionTimer;

    #[test]
    fn fused_cycles_are_reported_without_segmented_tokens() {
        let timer = GpuSectionTimer::default();
        assert!(timer.report().is_none());
        timer.record_mtp_cycle(0, 99);
        assert!(timer.report().is_none());
        timer.record_mtp_cycle(1, 2_000_000);
        timer.record_mtp_cycle(1, 4_000_000);
        timer.record_mtp_cycle(2, 5_000_000);
        let report = timer.report().expect("invariant: cycles enregistrés");
        assert!(report.contains("D1 (n=2"));
        assert!(report.contains("3.000 ms/cycle"));
        assert!(report.contains("D2 (n=1"));
        assert!(!report.contains("ms/token"));
        timer.record_layer(true, 1_000_000);
        timer.record_lmhead(1_000_000);
        let mixed = timer.report().expect("invariant: token enregistré");
        assert!(mixed.contains("sections/token (n=1"));
        assert!(mixed.contains(&report));
        timer.record_mtp_flush(7_000_000);
        let flushed = timer.report().expect("invariant: flush enregistré");
        assert!(flushed.contains("7.000 ms/flush"));
        assert!(flushed.contains(&mixed));
        timer.record_mtp_d2_segments(super::MtpD2Segments {
            append_ns: 1_000_000,
            head_ns: 2_000_000,
            verify_full_ns: 4_000_000,
            verify_full_layers: 2,
            verify_linear_ns: 3_000_000,
            verify_linear_layers: 3,
            tail_ns: 1_000_000,
            cycles: 1,
        });
        let segmented = timer.report().expect("invariant: segments enregistrés");
        assert!(
            segmented.contains("D2 (n=2"),
            "le total des segments compte un cycle D2"
        );
        assert!(segmented.contains("8.000 ms/cycle"), "(5 + 11) / 2");
        assert!(segmented.contains("tête 2.000 ms"));
        assert!(segmented.contains("verify full-attn 4.000 ms (2.000 ms/couche ×2)"));
        assert!(segmented.contains("verify linear-attn 3.000 ms (1.000 ms/couche ×3)"));
    }
}
