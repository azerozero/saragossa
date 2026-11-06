//! Chargement GPU des poids Flash-Next depuis un checkpoint q4e8.
//!
//! Les triplets affines sont lus bruts (sans déquantification) puis copiés
//! dans des buffers Metal ; une couche à la fois, ses copies CPU temporaires
//! sont libérées avant la suivante. Les petits tenseurs denses (routeur,
//! injections, projections `a`/`b` du DeltaNet, normes) restent vivants côté
//! CPU : le cache de l'exécuteur indexe ces buffers par adresse, une adresse
//! recyclée par une autre couche renverrait sinon un buffer périmé.

use metal::{Buffer, MTLResourceOptions};

use super::attention::QsaLayerWeights;
use super::hc::HcWeights;
use super::ple::PleWeights;
use crate::metal_backend::{
    LinearAttnResidentWeights, MetalExecutor, MetalLinearAttnResidentDenseWeights,
    MetalLinearWeightBuffers, MetalMoeSharedWeights,
};
use crate::qwen_loader::qwen4_exp::{Qwen4ExpCheckpoint, Qwen4ExpLayerKind};
use crate::{GatedMlp, InferError, Linear, LinearWeight, MemoryGuard, Result, Tensor};

/// Mélangeur de séquence d'une couche.
pub(crate) enum MixerWeights {
    Linear(Box<MetalLinearAttnResidentDenseWeights>),
    Full(Box<QsaLayerWeights>),
}

/// Poids GPU d'une couche du tronc.
pub(crate) struct LayerWeights {
    pub(crate) attn_hc: HcWeights,
    pub(crate) mlp_hc: HcWeights,
    pub(crate) mixer: MixerWeights,
    pub(crate) moe: MetalMoeSharedWeights,
}

/// Poids GPU du tronc (sans MTP) et sources CPU maintenues en vie.
pub(crate) struct FlashNextWeights {
    pub(crate) layers: Vec<LayerWeights>,
    pub(crate) mixer: HcWeights,
    pub(crate) lm_head: MetalLinearWeightBuffers,
    pub(crate) ple: PleWeights,
    /// Sources des buffers mis en cache par adresse (voir l'en-tête).
    keepalive: Vec<Linear>,
    keepalive_tensors: Vec<Tensor>,
    keepalive_vectors: Vec<Vec<f32>>,
}

fn f32_buffer(executor: &MetalExecutor, data: &[f32]) -> Buffer {
    executor.device().new_buffer_with_data(
        data.as_ptr().cast(),
        std::mem::size_of_val(data) as u64,
        MTLResourceOptions::StorageModeShared,
    )
}

fn quantized(ckpt: &Qwen4ExpCheckpoint, base: &str, stack: Option<usize>) -> Result<Linear> {
    Linear::from_weight(
        LinearWeight::AffineQuantized(ckpt.quantized(base, stack)?),
        None,
    )
}

fn dense(ckpt: &Qwen4ExpCheckpoint, name: &str, rows: usize, cols: usize) -> Result<Linear> {
    Linear::new(
        Tensor::from_vec(vec![rows, cols], ckpt.floats(name)?)?,
        None,
    )
}

fn one_plus(values: Vec<f32>) -> Vec<f32> {
    values.into_iter().map(|w| 1.0 + w).collect()
}

/// Octets d'un checkpoint attendus en mémoire GPU pour une couche : somme des
/// tenseurs du préfixe (garde mémoire, au chargement).
fn layer_bytes(ckpt: &Qwen4ExpCheckpoint, prefix: &str) -> u64 {
    ckpt.tensor_bytes_with_prefix(prefix)
}

impl FlashNextWeights {
    /// Charge le tronc sur GPU, couche par couche, sous la garde mémoire.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur de lecture, de forme, d'upload Metal, ou
    /// [`InferError::MemoryGuard`] si une couche dépasserait le plafond.
    pub(crate) fn load(
        executor: &MetalExecutor,
        ckpt: &Qwen4ExpCheckpoint,
        guard: Option<&MemoryGuard>,
    ) -> Result<Self> {
        let c = &ckpt.config;
        super::session::validate_resident_config(c)?;
        let mut weights = Self {
            layers: Vec::with_capacity(c.layer_kinds.len()),
            mixer: Self::hc(
                executor,
                ckpt,
                &Qwen4ExpCheckpoint::trunk("hyper_connection_mixer."),
                false,
            )?,
            lm_head: executor.resolve_linear_weight_buffers(
                &LinearWeight::AffineQuantized(
                    ckpt.quantized(Qwen4ExpCheckpoint::lm_head(), None)?,
                ),
                "flash_next_lm_head",
            )?,
            ple: Self::ple(executor, ckpt)?,
            keepalive: Vec::new(),
            keepalive_tensors: Vec::new(),
            keepalive_vectors: Vec::new(),
        };
        for (index, kind) in c.layer_kinds.iter().enumerate() {
            let prefix = Qwen4ExpCheckpoint::layer_prefix(index);
            if let Some(guard) = guard {
                guard
                    .check_allocation(layer_bytes(ckpt, &prefix))
                    .map_err(InferError::MemoryGuard)?;
            }
            let layer = weights.layer(executor, ckpt, &prefix, *kind)?;
            weights.layers.push(layer);
        }
        Ok(weights)
    }

    fn hc(
        executor: &MetalExecutor,
        ckpt: &Qwen4ExpCheckpoint,
        prefix: &str,
        inject: bool,
    ) -> Result<HcWeights> {
        let c = &ckpt.config;
        let resolve = |base: String| -> Result<MetalLinearWeightBuffers> {
            executor.resolve_linear_weight_buffers(
                &LinearWeight::AffineQuantized(ckpt.quantized(&base, None)?),
                "flash_next_hc",
            )
        };
        let inject = if inject {
            // Buffer propre (bf16 → f32) : pas de cache par adresse.
            let data = ckpt.floats(&format!("{prefix}block_inject_weight.weight"))?;
            Some(MetalLinearWeightBuffers::Dense {
                rhs: f32_buffer(executor, &data),
                rhs_bf16: None,
                out_dim: c.hc_count,
                in_dim: c.hc_width(),
            })
        } else {
            None
        };
        Ok(HcWeights {
            norm: f32_buffer(executor, &ckpt.floats(&format!("{prefix}hc_norm.weight"))?),
            down: resolve(format!("{prefix}input_mix_weight_down"))?,
            up: resolve(format!("{prefix}input_mix_weight_up"))?,
            inject,
        })
    }

    fn ple(executor: &MetalExecutor, ckpt: &Qwen4ExpCheckpoint) -> Result<PleWeights> {
        let c = &ckpt.config;
        let e = format!("{}ple.", Qwen4ExpCheckpoint::layer_prefix(c.ple_layer));
        let resolve = |base: String| -> Result<MetalLinearWeightBuffers> {
            executor.resolve_linear_weight_buffers(
                &LinearWeight::AffineQuantized(ckpt.quantized(&base, None)?),
                "flash_next_ple",
            )
        };
        let floats = |name: &str| -> Result<Buffer> {
            Ok(f32_buffer(executor, &ckpt.floats(&format!("{e}{name}"))?))
        };
        Ok(PleWeights {
            key_proj: resolve(format!("{e}key_proj"))?,
            value_proj: resolve(format!("{e}value_proj"))?,
            norm_key: floats("norm_key.weight")?,
            norm_query: floats("norm_query.weight")?,
            norm_conv: floats("norm_conv.weight")?,
            conv: floats("conv1d.weight")?,
            lut: f32_buffer(executor, &ckpt.ngram_lut()),
            kernel: c.ple_conv_kernel,
            dilation: c.ngram_size,
            ple_dim: c.ple_dim,
        })
    }

    fn layer(
        &mut self,
        executor: &MetalExecutor,
        ckpt: &Qwen4ExpCheckpoint,
        prefix: &str,
        kind: Qwen4ExpLayerKind,
    ) -> Result<LayerWeights> {
        let c = &ckpt.config;
        let attn_hc = Self::hc(
            executor,
            ckpt,
            &format!("{prefix}attn_hyper_connection."),
            true,
        )?;
        let mlp_hc = Self::hc(
            executor,
            ckpt,
            &format!("{prefix}mlp_hyper_connection."),
            true,
        )?;
        let mixer = match kind {
            Qwen4ExpLayerKind::Linear => {
                MixerWeights::Linear(Box::new(self.deltanet(executor, ckpt, prefix)?))
            }
            Qwen4ExpLayerKind::Full => {
                let a = format!("{prefix}self_attn.");
                let resolve = |name: &str| -> Result<MetalLinearWeightBuffers> {
                    executor.resolve_linear_weight_buffers(
                        &LinearWeight::AffineQuantized(
                            ckpt.quantized(&format!("{a}{name}"), None)?,
                        ),
                        "flash_next_attn",
                    )
                };
                MixerWeights::Full(Box::new(QsaLayerWeights {
                    q_proj: resolve("q_proj")?,
                    k_proj: resolve("k_proj")?,
                    v_proj: resolve("v_proj")?,
                    o_proj: resolve("o_proj")?,
                    q_norm: f32_buffer(
                        executor,
                        &one_plus(ckpt.floats(&format!("{a}q_norm.weight"))?),
                    ),
                    k_norm: f32_buffer(
                        executor,
                        &one_plus(ckpt.floats(&format!("{a}k_norm.weight"))?),
                    ),
                }))
            }
        };
        let m = format!("{prefix}mlp.");
        let experts: Vec<GatedMlp> = (0..c.experts)
            .map(|e| {
                Ok(GatedMlp::new(
                    quantized(ckpt, &format!("{m}switch_mlp.gate_proj"), Some(e))?,
                    quantized(ckpt, &format!("{m}switch_mlp.up_proj"), Some(e))?,
                    quantized(ckpt, &format!("{m}switch_mlp.down_proj"), Some(e))?,
                ))
            })
            .collect::<Result<_>>()?;
        let shared = GatedMlp::new(
            quantized(ckpt, &format!("{m}shared_expert.gate_proj"), None)?,
            quantized(ckpt, &format!("{m}shared_expert.up_proj"), None)?,
            quantized(ckpt, &format!("{m}shared_expert.down_proj"), None)?,
        );
        let router = dense(ckpt, &format!("{m}gate.weight"), c.experts, c.hidden)?;
        let shared_gate = dense(ckpt, &format!("{m}shared_expert_gate.weight"), 1, c.hidden)?;
        let moe = executor.resolve_moe_shared_weights(&router, &experts, &shared, &shared_gate)?;
        self.keepalive.extend([router, shared_gate]);
        Ok(LayerWeights {
            attn_hc,
            mlp_hc,
            mixer,
            moe,
        })
    }

    fn deltanet(
        &mut self,
        executor: &MetalExecutor,
        ckpt: &Qwen4ExpCheckpoint,
        prefix: &str,
    ) -> Result<MetalLinearAttnResidentDenseWeights> {
        let c = &ckpt.config;
        let l = format!("{prefix}linear_attn.");
        let conv_dim =
            2 * c.linear_key_heads * c.linear_key_dim + c.linear_value_heads * c.linear_value_dim;
        let in_qkv = quantized(ckpt, &format!("{l}in_proj_qkv"), None)?;
        let in_z = quantized(ckpt, &format!("{l}in_proj_z"), None)?;
        let in_b = dense(
            ckpt,
            &format!("{l}in_proj_b.weight"),
            c.linear_value_heads,
            c.hidden,
        )?;
        let in_a = dense(
            ckpt,
            &format!("{l}in_proj_a.weight"),
            c.linear_value_heads,
            c.hidden,
        )?;
        let out = quantized(ckpt, &format!("{l}out_proj"), None)?;
        let conv = Tensor::from_vec(
            vec![conv_dim, 1, c.conv_kernel],
            ckpt.floats(&format!("{l}conv1d.weight"))?,
        )?;
        let a_log = ckpt.floats(&format!("{l}A_log"))?;
        let dt_bias = ckpt.floats(&format!("{l}dt_bias"))?;
        // Gain direct `w` (Qwen4ExpTextRMSNormGated), sans ajout de 1.
        let norm = ckpt.floats(&format!("{l}norm.weight"))?;
        let resolved =
            executor.resolve_linear_attn_resident_dense_weights(LinearAttnResidentWeights {
                in_proj_qkv: &in_qkv,
                in_proj_z: &in_z,
                in_proj_b: &in_b,
                in_proj_a: &in_a,
                out_proj: &out,
                conv_weight: &conv,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm_weight: &norm,
            })?;
        self.keepalive.extend([in_b, in_a]);
        self.keepalive_tensors.push(conv);
        self.keepalive_vectors.extend([a_log, dt_bias, norm]);
        Ok(resolved)
    }
}
