//! Types de poids et de profilage du decode résident.

use super::*;

/// Poids d'UNE couche full-attn MoE (decode résident 1c) : handles Metal résolus
/// au setup de génération, sans lookup de cache dans le chemin par token.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub qkv_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub moe: &'a MetalMoeSharedWeights,
    pub top_k: usize,
}

/// Poids d'UNE couche full-attn MoE routed-only résidente.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnRoutedLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub qkv_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub moe: &'a MetalMoeRoutedWeights,
    pub top_k: usize,
}

/// Poids d'UNE couche full-attn dense (decode résident 1c). Structure séparée
/// du chemin MoE pour garder les préconditions et les poids MoE inchangés.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnDenseLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub qkv_proj: Option<&'a MetalLinearWeightBuffers>,
    pub q_proj: &'a MetalLinearWeightBuffers,
    pub k_proj: &'a MetalLinearWeightBuffers,
    pub v_proj: &'a MetalLinearWeightBuffers,
    pub o_proj: &'a MetalLinearWeightBuffers,
    pub q_norm: &'a Buffer,
    pub k_norm: &'a Buffer,
    pub post_norm: &'a Buffer,
    pub gate_proj: &'a MetalLinearWeightBuffers,
    pub up_proj: &'a MetalLinearWeightBuffers,
    pub down_proj: &'a MetalLinearWeightBuffers,
    pub tail_score: &'a Buffer,
}

/// Dimensions + paramètres RoPE d'UNE couche full-attn de decode résident.
#[derive(Clone, Copy)]
pub(crate) struct FullAttnLayerDims {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rope_dims: usize,
    pub position: usize,
    pub eps: f32,
    pub theta: f32,
    pub attn_output_gate: bool,
}

/// Poids d'UNE couche linear-attn MoE (decode résident 1c) : handles Metal
/// résolus au setup de génération, sans lookup de cache dans le chemin par token.
#[derive(Clone, Copy)]
pub(crate) struct LinearAttnLayerWeights<'a> {
    pub input_norm: &'a Buffer,
    pub linear: &'a MetalLinearAttnResidentWeights,
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

/// Cumul CPU des temps par section du decode résident (ÉTAPE 0 tranche 3,
/// fallback `RETI_RUST_GPU_COUNTERS`). Sur Apple Silicon (AGX) l'échantillonnage
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
}

impl GpuSectionTimer {
    /// Construit le timer si `RETI_RUST_GPU_COUNTERS` est défini, sinon `None`
    /// (le decode reste le chemin résident à command buffer unique, inchangé).
    pub(crate) fn try_new() -> Option<Self> {
        crate::decoder::flags::gpu_counters_enabled().then_some(())?;
        eprintln!("gpu sections: actif (segmentation CPU par section)");
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

    /// Formate le classement per-section (ms/token + %), ou `None` si rien
    /// d'accumulé. Donne les moyennes par couche full vs linear et le total lm_head.
    pub(crate) fn report(&self) -> Option<String> {
        let accum = *self.accum.borrow();
        if accum.tokens == 0 {
            return None;
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
             lm_head {lmhead:.3} ms ({lmhead_p:.1}%) | Σ {sum:.3} ms/token",
            tok = accum.tokens,
            nf = accum.full_layer_count / accum.tokens,
            nl = accum.linear_layer_count / accum.tokens,
            full_p = pct(full_total),
            linear_p = pct(linear_total),
            lmhead_p = pct(lmhead),
        ))
    }
}
