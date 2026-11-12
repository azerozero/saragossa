//! Chargement et forward de la tête MTP.
//!
//! Supporte le format **OptiQ int4 + MLP dense** (27B-OptiQ : `.weight` u32 packé
//! 4-bit gs64 + `.scales`/`.biases` bf16, MLP `gate/up/down_proj`) en plus des
//! normes/`fc` bf16 (27B), et la tête **officielle transformers à MLP MoE**
//! (35B-A3B : experts fusionnés `gate_up_proj`, bf16), quantifiée à la charge
//! en affine gs64 comme le tronc, normes en `1 + weight`.

use super::*;
use crate::quantization::bytes_to_u32;
use crate::safetensor::{bytes_to_dense_f32, pread_exact};
use crate::AffineQuantizedTensor;
#[cfg(all(target_os = "macos", feature = "metal"))]
use safetensors::Dtype;
use safetensors::{tensor::TensorView, SafeTensors};
use std::collections::HashMap;

pub(super) struct MtpLoadContext<'a> {
    #[cfg(all(target_os = "macos", feature = "metal"))]
    metal: Option<&'a crate::MetalExecutor>,
    marker: std::marker::PhantomData<&'a ()>,
}

impl MtpLoadContext<'_> {
    pub(super) fn cpu() -> Self {
        Self {
            #[cfg(all(target_os = "macos", feature = "metal"))]
            metal: None,
            marker: std::marker::PhantomData,
        }
    }

    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn metal(executor: &crate::MetalExecutor) -> MtpLoadContext<'_> {
        MtpLoadContext {
            metal: Some(executor),
            marker: std::marker::PhantomData,
        }
    }
}

/// Convention de stockage des normes de la tête.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MtpNormConvention {
    /// Sidecars 27B (mlx-community, OptiQ) : décalage détecté par la moyenne,
    /// chemin historique conservé tel quel.
    MeanHeuristic,
    /// Checkpoint transformers officiel Qwen3.5/3.6 : chaque `RMSNorm` applique
    /// `1 + weight`, quelle que soit la moyenne stockée (une norme finale
    /// stockée à 1,9 vaut 2,9) ; l'heuristique en manquerait quatre sur sept.
    UnitOffset,
}

/// Quantification appliquée à la charge d'une tête MoE bf16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum MtpMoeQuant {
    /// Affine gs64 comme le tronc mlx-community : u4, routeur et gate de
    /// l'expert partagé en u8. Seul format accepté par le MoE résident.
    Affine,
    /// Poids f32 non quantifiés : oracle CPU de la perte de quantification.
    #[cfg(test)]
    Dense,
}

const MTP_DENSE_MARKER: &str = "mtp.layers.0.mlp.gate_proj.weight";
const MTP_MOE_MARKER: &str = "mtp.layers.0.mlp.experts.gate_up_proj";
/// `down_proj` des experts fusionnés. Avec `MTP_MOE_MARKER`, seuls tenseurs
/// laissés sur disque jusqu'à leur quantification (`LazySidecar`).
const MTP_MOE_DOWN: &str = "mtp.layers.0.mlp.experts.down_proj";
const MTP_QUANT_GROUP: usize = 64;

impl MtpHead {
    /// Déplace les poids quantifiés dans des `MTLBuffer` partagés, experts en
    /// stacks contigus (`contiguous_metal_expert_stacks`) : une seule copie.
    #[cfg(all(target_os = "macos", feature = "metal"))]
    pub(super) fn move_packed_to_metal_shared(
        &mut self,
        metal: &crate::MetalExecutor,
    ) -> Result<()> {
        self.fc.move_packed_to_metal_shared(metal)?;
        self.layer.attention.move_packed_to_metal_shared(metal)?;
        self.layer.mlp.move_packed_to_metal_shared(metal)
    }

    pub(super) fn from_sidecar(
        path: impl AsRef<Path>,
        config: &CausalDecoderConfig,
        hidden: usize,
        context: &MtpLoadContext<'_>,
    ) -> Result<Self> {
        Self::from_sidecar_with(path, config, hidden, context, MtpMoeQuant::Affine)
    }

    /// `hidden` est la largeur du tronc (norme finale) : toutes les formes de la
    /// tête sont validées contre elle et la config avant tout encodage GPU,
    /// dont les kernels lisent `hidden` coefficients sans borner le buffer.
    pub(super) fn from_sidecar_with(
        path: impl AsRef<Path>,
        config: &CausalDecoderConfig,
        hidden: usize,
        context: &MtpLoadContext<'_>,
        moe_quant: MtpMoeQuant,
    ) -> Result<Self> {
        let sidecar = LazySidecar::open(path.as_ref())?;
        let st = SafeTensors::deserialize(&sidecar.bytes)
            .map_err(|source| InferError::Config(format!("parse sidecar MTP: {source}")))?;
        // Résout les clés normalisées (sans préfixe `language_model.`) → clé réelle.
        let mut resolve: HashMap<String, String> = HashMap::new();
        for name in st.names() {
            let normalized = normalize_mtp_sidecar_key(name);
            if let Some(previous) = resolve.insert(normalized.clone(), name.to_string()) {
                // Deux clés réelles pour une même clé normalisée : laquelle
                // charger serait arbitraire.
                return Err(InferError::Config(format!(
                    "sidecar MTP : `{previous}` et `{name}` se normalisent en `{normalized}`"
                )));
            }
        }
        let q_rows = st_view(&st, &resolve, "mtp.layers.0.self_attn.q_proj.weight")?
            .shape()
            .first()
            .copied()
            .unwrap_or(0);
        let dims = MtpDims::new(config, hidden, q_rows)?;
        match (
            resolve.contains_key(MTP_DENSE_MARKER),
            resolve.contains_key(MTP_MOE_MARKER),
        ) {
            (true, false) => Self::dense_from_tensors(&st, &resolve, config, &dims, context),
            (false, true) => {
                Self::moe_from_tensors(&st, &sidecar, &resolve, config, &dims, context, moe_quant)
            }
            (true, true) => Err(InferError::Config(
                "sidecar MTP ambigu : MLP dense et experts MoE présents".to_string(),
            )),
            (false, false) => Err(InferError::MissingWeight(format!(
                "{MTP_DENSE_MARKER} ou {MTP_MOE_MARKER}"
            ))),
        }
    }

    /// Tête 27B : normes, attention et MLP dense, quantifiés ou bf16 tels que
    /// stockés (chemin historique inchangé).
    fn dense_from_tensors(
        st: &SafeTensors,
        resolve: &HashMap<String, String>,
        config: &CausalDecoderConfig,
        dims: &MtpDims,
        context: &MtpLoadContext<'_>,
    ) -> Result<Self> {
        let norms = MtpNorms::load(st, resolve, MtpNormConvention::MeanHeuristic)?;
        let fc = load_mtp_linear(st, resolve, "mtp.fc", context)?;
        let attention = mtp_attention(
            [
                load_mtp_linear(st, resolve, "mtp.layers.0.self_attn.q_proj", context)?,
                load_mtp_linear(st, resolve, "mtp.layers.0.self_attn.k_proj", context)?,
                load_mtp_linear(st, resolve, "mtp.layers.0.self_attn.v_proj", context)?,
                load_mtp_linear(st, resolve, "mtp.layers.0.self_attn.o_proj", context)?,
            ],
            norms.q_norm.clone(),
            norms.k_norm.clone(),
            config,
        );
        let mlp = GatedMlp::new(
            load_mtp_linear(st, resolve, "mtp.layers.0.mlp.gate_proj", context)?,
            load_mtp_linear(st, resolve, "mtp.layers.0.mlp.up_proj", context)?,
            load_mtp_linear(st, resolve, "mtp.layers.0.mlp.down_proj", context)?,
        );
        dims.check_gated_mlp(&mlp, "mtp.layers.0.mlp")?;
        let head = norms.into_head(fc, attention, FeedForward::Dense(Box::new(mlp)));
        dims.check_common(&head)?;
        Ok(head)
    }

    /// Tête officielle Qwen3.6-35B-A3B : attention complète avec porte de
    /// sortie et MoE softmax top-k à expert partagé gated, tout en bf16.
    fn moe_from_tensors(
        st: &SafeTensors,
        sidecar: &LazySidecar,
        resolve: &HashMap<String, String>,
        config: &CausalDecoderConfig,
        dims: &MtpDims,
        context: &MtpLoadContext<'_>,
        quant: MtpMoeQuant,
    ) -> Result<Self> {
        if !config.attn_output_gate {
            // `encode_full_attn_layer` (queue MoE résidente) suppose une Q à porte.
            return Err(InferError::Config(
                "tête MTP MoE : attn_output_gate requis".to_string(),
            ));
        }
        let norms = MtpNorms::load(st, resolve, MtpNormConvention::UnitOffset)?;
        let linear = |prefix: &str, bits: usize| -> Result<Linear> {
            quantize_mtp_linear(load_mtp_linear(st, resolve, prefix, context)?, quant, bits)
        };
        // fc et attention en u8 : +18 Mio par draft (≈ 40 µs) contre 1,5 point
        // d'accord perdu en u4 (porte G1, 1200 positions) ; experts et
        // expert partagé en u4, format du stack MoE résident.
        let fc = linear("mtp.fc", 8)?;
        let attention = mtp_attention(
            [
                linear("mtp.layers.0.self_attn.q_proj", 8)?,
                linear("mtp.layers.0.self_attn.k_proj", 8)?,
                linear("mtp.layers.0.self_attn.v_proj", 8)?,
                linear("mtp.layers.0.self_attn.o_proj", 8)?,
            ],
            norms.q_norm.clone(),
            norms.k_norm.clone(),
            config,
        );
        let (gate_weights, up_weights, down_weights) =
            load_mtp_fused_experts(st, sidecar, resolve, dims.hidden, quant, context)?;
        let expert_count = gate_weights.len();
        if config
            .num_experts
            .is_some_and(|expected| expected != expert_count)
        {
            return Err(InferError::Dimension(format!(
                "tête MTP MoE : {expert_count} experts, config {:?}",
                config.num_experts
            )));
        }
        let top_k = config.num_experts_per_tok;
        if top_k == 0 || top_k > expert_count {
            return Err(InferError::Config(format!(
                "tête MTP MoE : top_k={top_k} hors de 1..={expert_count}"
            )));
        }
        let experts = split_experts(gate_weights, up_weights, down_weights, expert_count)?;
        let shared_expert = GatedMlp::new(
            linear("mtp.layers.0.mlp.shared_expert.gate_proj", 4)?,
            linear("mtp.layers.0.mlp.shared_expert.up_proj", 4)?,
            linear("mtp.layers.0.mlp.shared_expert.down_proj", 4)?,
        );
        dims.check_gated_mlp(&shared_expert, "mtp.layers.0.mlp.shared_expert")?;
        // Routeur et gate partagée en u8, comme dans le tronc mlx-community.
        let router = linear("mtp.layers.0.mlp.gate", 8)?;
        expect_shape(
            &router,
            [expert_count, dims.hidden],
            "mtp.layers.0.mlp.gate",
        )?;
        let shared_gate = linear("mtp.layers.0.mlp.shared_expert_gate", 8)?;
        expect_shape(
            &shared_gate,
            [1, dims.hidden],
            "mtp.layers.0.mlp.shared_expert_gate",
        )?;
        let moe = crate::MoeMlp::new(
            router,
            experts,
            Some(shared_expert),
            Some(shared_gate),
            top_k,
        )?
        .with_router_score(config.router_score);
        let head = norms.into_head(fc, attention, FeedForward::Moe(Box::new(moe)));
        dims.check_common(&head)?;
        Ok(head)
    }
}

/// Formes attendues de la tête, tirées du tronc et de sa config.
pub(super) struct MtpDims {
    hidden: usize,
    head_dim: usize,
    q_out: usize,
    kv_out: usize,
    o_in: usize,
}

impl MtpDims {
    /// `q_rows` : sorties de `q_proj` dans le sidecar. Sans `head_dim` dans la
    /// config, il en est déduit comme pour l'attention du tronc
    /// (`infer_attention_head_dim`).
    fn new(config: &CausalDecoderConfig, hidden: usize, q_rows: usize) -> Result<Self> {
        let gate = if config.attn_output_gate { 2 } else { 1 };
        let head_dim = match config.head_dim {
            Some(head_dim) => head_dim,
            None => {
                let divisor = config.num_attention_heads * gate;
                if divisor == 0 || q_rows % divisor != 0 {
                    return Err(InferError::Dimension(format!(
                        "tête MTP : q_proj {q_rows} sorties, incompatible avec {} têtes (porte={})",
                        config.num_attention_heads, config.attn_output_gate
                    )));
                }
                q_rows / divisor
            }
        };
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| InferError::Dimension("tête MTP : dimension déborde".into()))
        };
        let o_in = product(config.num_attention_heads, head_dim)?;
        // Q porte la gate de sortie concaténée par tête quand elle existe.
        let q_out = product(o_in, gate)?;
        Ok(Self {
            hidden,
            head_dim,
            q_out,
            kv_out: product(config.num_key_value_heads, head_dim)?,
            o_in,
        })
    }

    /// Normes, `fc` et attention, communes aux têtes dense et MoE.
    fn check_common(&self, head: &MtpHead) -> Result<()> {
        let hidden = self.hidden;
        for (norm, name) in [
            (&head.pre_fc_norm_embedding, "mtp.pre_fc_norm_embedding"),
            (&head.pre_fc_norm_hidden, "mtp.pre_fc_norm_hidden"),
            (&head.layer.input_norm, "mtp.layers.0.input_layernorm"),
            (
                &head.layer.post_attention_norm,
                "mtp.layers.0.post_attention_layernorm",
            ),
            (&head.norm, "mtp.norm"),
        ] {
            expect_len(norm, hidden, name)?;
        }
        let attention = &head.layer.attention;
        for (norm, name) in [
            (&attention.q_norm, "mtp.layers.0.self_attn.q_norm"),
            (&attention.k_norm, "mtp.layers.0.self_attn.k_norm"),
        ] {
            let norm = norm
                .as_ref()
                .ok_or_else(|| InferError::MissingWeight(name.to_string()))?;
            expect_len(norm, self.head_dim, name)?;
        }
        let double = hidden
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("tête MTP : fc déborde".into()))?;
        expect_shape(&head.fc, [hidden, double], "mtp.fc")?;
        expect_shape(
            &attention.q_proj,
            [self.q_out, hidden],
            "mtp.layers.0.self_attn.q_proj",
        )?;
        expect_shape(
            &attention.k_proj,
            [self.kv_out, hidden],
            "mtp.layers.0.self_attn.k_proj",
        )?;
        let v_proj = attention
            .v_proj
            .as_ref()
            .ok_or_else(|| InferError::MissingWeight("mtp.layers.0.self_attn.v_proj".into()))?;
        expect_shape(
            v_proj,
            [self.kv_out, hidden],
            "mtp.layers.0.self_attn.v_proj",
        )?;
        expect_shape(
            &attention.o_proj,
            [hidden, self.o_in],
            "mtp.layers.0.self_attn.o_proj",
        )
    }

    /// MLP SwiGLU : gate et up `[I, hidden]`, down `[hidden, I]`.
    fn check_gated_mlp(&self, mlp: &GatedMlp, prefix: &str) -> Result<()> {
        let (gate, up, down) = mlp.projections();
        let inter = gate.weight().shape().first().copied().unwrap_or(0);
        expect_shape(gate, [inter, self.hidden], &format!("{prefix}.gate_proj"))?;
        expect_shape(up, [inter, self.hidden], &format!("{prefix}.up_proj"))?;
        expect_shape(down, [self.hidden, inter], &format!("{prefix}.down_proj"))
    }
}

fn expect_shape(linear: &Linear, expected: [usize; 2], name: &str) -> Result<()> {
    let shape = linear.weight().shape();
    if shape != expected.as_slice() || expected.contains(&0) {
        return Err(InferError::Dimension(format!(
            "tête MTP {name} : forme {shape:?}, attendu {expected:?}"
        )));
    }
    Ok(())
}

fn expect_len(norm: &Tensor, len: usize, name: &str) -> Result<()> {
    if norm.shape() != [len] {
        return Err(InferError::Dimension(format!(
            "tête MTP {name} : forme {:?}, attendu [{len}]",
            norm.shape()
        )));
    }
    Ok(())
}

/// Normes de la tête, chargées selon une même convention.
struct MtpNorms {
    pre_fc_norm_embedding: Tensor,
    pre_fc_norm_hidden: Tensor,
    input_norm: Tensor,
    q_norm: Tensor,
    k_norm: Tensor,
    post_attention_norm: Tensor,
    norm: Tensor,
}

impl MtpNorms {
    fn load(
        st: &SafeTensors,
        resolve: &HashMap<String, String>,
        convention: MtpNormConvention,
    ) -> Result<Self> {
        let norm = |key: &str| load_mtp_norm(st, resolve, key, convention);
        Ok(Self {
            pre_fc_norm_embedding: norm("mtp.pre_fc_norm_embedding.weight")?,
            pre_fc_norm_hidden: norm("mtp.pre_fc_norm_hidden.weight")?,
            input_norm: norm("mtp.layers.0.input_layernorm.weight")?,
            q_norm: norm("mtp.layers.0.self_attn.q_norm.weight")?,
            k_norm: norm("mtp.layers.0.self_attn.k_norm.weight")?,
            post_attention_norm: norm("mtp.layers.0.post_attention_layernorm.weight")?,
            norm: norm("mtp.norm.weight")?,
        })
    }

    fn into_head(self, fc: Linear, attention: FullAttention, mlp: FeedForward) -> MtpHead {
        MtpHead {
            pre_fc_norm_embedding: self.pre_fc_norm_embedding,
            pre_fc_norm_hidden: self.pre_fc_norm_hidden,
            fc,
            layer: MtpLayer {
                input_norm: self.input_norm,
                attention,
                post_attention_norm: self.post_attention_norm,
                mlp,
            },
            norm: self.norm,
        }
    }
}

fn mtp_attention(
    [q_proj, k_proj, v_proj, o_proj]: [Linear; 4],
    q_norm: Tensor,
    k_norm: Tensor,
    config: &CausalDecoderConfig,
) -> FullAttention {
    FullAttention {
        q_proj,
        k_proj,
        v_proj: Some(v_proj),
        o_proj,
        q_norm: Some(q_norm),
        k_norm: Some(k_norm),
        num_key_value_heads: None,
        head_dim: None,
        // RoPE partielle : fréquences sur les `rope_dims` tournées, comme les
        // couches du tronc et le decode résident. Sans ce réglage, le forward
        // CPU et le semis de l'historique retombaient sur `head_dim` (256 au
        // lieu de 64 sur Qwen3.5/3.6) et divergeaient du draft résident.
        rope_dims: config.rope_dims,
        rope_frequency_dim: config.rope_dims,
        value_norm: false,
        // MTP = sidecar Qwen, base RoPE unique de la config, positions
        // brutes, attention pleine.
        rope_theta: None,
        rope_position_scale: None,
        sliding_window: None,
        sinks: None,
        no_rope: false,
    }
}

/// Quantifie à la charge une linéaire dense de la tête MoE. Une linéaire déjà
/// quantifiée dans le sidecar est gardée telle quelle.
fn quantize_mtp_linear(linear: Linear, quant: MtpMoeQuant, bits: usize) -> Result<Linear> {
    match (quant, linear.weight()) {
        (MtpMoeQuant::Affine, LinearWeight::Dense(dense)) => Linear::from_weight(
            LinearWeight::AffineQuantized(AffineQuantizedTensor::quantize(
                dense,
                MTP_QUANT_GROUP,
                bits,
            )?),
            None,
        ),
        _ => Ok(linear),
    }
}

/// Experts fusionnés au format transformers : `gate_up_proj` `[E, 2I, H]`
/// (sortie découpée par `chunk(2)` : gate = lignes `0..I`, up = `I..2I`) et
/// `down_proj` `[E, H, I]`, sans suffixe `.weight`. Convertis expert par
/// expert pour ne jamais tenir les 256 experts en f32 (≈ 3,2 Go).
#[allow(
    clippy::type_complexity,
    reason = "triplet gate/up/down consommé immédiatement par split_experts"
)]
fn load_mtp_fused_experts(
    st: &SafeTensors,
    sidecar: &LazySidecar,
    resolve: &HashMap<String, String>,
    expected_hidden: usize,
    quant: MtpMoeQuant,
    #[cfg_attr(
        not(all(target_os = "macos", feature = "metal")),
        expect(unused_variables, reason = "quantification Metal seulement")
    )]
    context: &MtpLoadContext<'_>,
) -> Result<(Vec<LinearWeight>, Vec<LinearWeight>, Vec<LinearWeight>)> {
    let gate_up = st_view(st, resolve, MTP_MOE_MARKER)?;
    let down_key = MTP_MOE_DOWN;
    let down = st_view(st, resolve, down_key)?;
    let (&[experts, two_inter, hidden], &[down_experts, down_hidden, inter]) =
        (gate_up.shape(), down.shape())
    else {
        return Err(InferError::Dimension(format!(
            "experts MTP : gate_up={:?}, down={:?}, rang 3 attendu",
            gate_up.shape(),
            down.shape()
        )));
    };
    if experts == 0
        || inter == 0
        || hidden != expected_hidden
        || Some(two_inter) != inter.checked_mul(2)
        || down_experts != experts
        || down_hidden != hidden
        || gate_up.data().len() % experts != 0
        || down.data().len() % experts != 0
    {
        return Err(InferError::Dimension(format!(
            "experts MTP incohérents : gate_up={:?}, down={:?}",
            gate_up.shape(),
            down.shape()
        )));
    }
    let weight = |tensor: Tensor| -> Result<LinearWeight> {
        match quant {
            MtpMoeQuant::Affine => Ok(LinearWeight::AffineQuantized(
                AffineQuantizedTensor::quantize(&tensor, MTP_QUANT_GROUP, 4)?,
            )),
            #[cfg(test)]
            MtpMoeQuant::Dense => Ok(LinearWeight::Dense(tensor)),
        }
    };
    let gate_up_stride = gate_up.data().len() / experts;
    let down_stride = down.data().len() / experts;
    let gate_up_offset = sidecar.data_offset(resolve, MTP_MOE_MARKER)?;
    let down_offset = sidecar.data_offset(resolve, down_key)?;
    #[cfg(all(target_os = "macos", feature = "metal"))]
    if let (MtpMoeQuant::Affine, Some(metal)) = (quant, context.metal) {
        if crate::runtime_flags::single_copy_weights_enabled() {
            // gate = première moitié de chaque expert `gate_up`, up = seconde.
            let half_bytes = gate_up_stride / 2;
            let projection =
                |offset: usize, stride: usize, rows: usize, cols: usize, dtype, key| {
                    quantize_experts_into_metal(
                        metal,
                        sidecar,
                        ExpertSlices {
                            offset,
                            stride,
                            rows,
                            cols,
                            dtype,
                            key,
                        },
                        experts,
                    )
                };
            return Ok((
                projection(
                    gate_up_offset,
                    gate_up_stride,
                    inter,
                    hidden,
                    gate_up.dtype(),
                    MTP_MOE_MARKER,
                )?,
                projection(
                    gate_up_offset + half_bytes,
                    gate_up_stride,
                    inter,
                    hidden,
                    gate_up.dtype(),
                    MTP_MOE_MARKER,
                )?,
                projection(
                    down_offset,
                    down_stride,
                    hidden,
                    inter,
                    down.dtype(),
                    down_key,
                )?,
            ));
        }
    }
    // Experts indépendants : quantifiés en parallèle (≈ 800 M poids), chacun lu
    // depuis le fichier, jamais les 1,6 Go d'experts bf16 d'un coup.
    let triples = (0..experts)
        .into_par_iter()
        .map(|expert| -> Result<_> {
            let bytes = sidecar.read(gate_up_offset + expert * gate_up_stride, gate_up_stride)?;
            let mut rows = bytes_to_dense_f32(&bytes, gate_up.dtype(), MTP_MOE_MARKER)?;
            // Garde explicite : `split_off` paniquerait sur un expert tronqué.
            let half = inter * hidden;
            if rows.len() != 2 * half {
                return Err(InferError::Dimension(format!(
                    "expert MTP {expert} : {} valeurs gate_up, attendu {}",
                    rows.len(),
                    2 * half
                )));
            }
            let up_rows = rows.split_off(half);
            let gate = weight(Tensor::from_vec(vec![inter, hidden], rows)?)?;
            let up = weight(Tensor::from_vec(vec![inter, hidden], up_rows)?)?;
            let bytes = sidecar.read(down_offset + expert * down_stride, down_stride)?;
            let rows = bytes_to_dense_f32(&bytes, down.dtype(), down_key)?;
            let down = weight(Tensor::from_vec(vec![hidden, inter], rows)?)?;
            Ok((gate, up, down))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut gates = Vec::with_capacity(experts);
    let mut ups = Vec::with_capacity(experts);
    let mut downs = Vec::with_capacity(experts);
    for (gate, up, down) in triples {
        gates.push(gate);
        ups.push(up);
        downs.push(down);
    }
    Ok((gates, ups, downs))
}

/// Une projection des experts dans le fichier : expert `e` à
/// `offset + e·stride`, matrice `[rows, cols]` de type `dtype`.
#[cfg(all(target_os = "macos", feature = "metal"))]
struct ExpertSlices<'a> {
    offset: usize,
    stride: usize,
    rows: usize,
    cols: usize,
    dtype: Dtype,
    key: &'a str,
}

/// Quantifie une projection des experts en u4 gs64 directement dans des
/// `MTLBuffer` partagés contigus, au format que `contiguous_metal_expert_stacks`
/// sert sans copie. Passer par des poids CPU puis `move_packed_to_metal_shared`
/// laissait ≈ 0,6 Go dans l'empreinte : malloc garde les blocs libérés.
#[cfg(all(target_os = "macos", feature = "metal"))]
fn quantize_experts_into_metal(
    metal: &crate::MetalExecutor,
    sidecar: &LazySidecar,
    slices: ExpertSlices<'_>,
    experts: usize,
) -> Result<Vec<LinearWeight>> {
    let ExpertSlices {
        offset,
        stride,
        rows,
        cols,
        dtype,
        key,
    } = slices;
    let (packed_cols, groups) = crate::quantization::affine_layout(rows, cols, MTP_QUANT_GROUP, 4)?;
    let packed_len = rows * packed_cols;
    let affine_len = rows * groups;
    let total = |len: usize| {
        len.checked_mul(experts)
            .ok_or_else(|| InferError::Shape(format!("experts MTP {key} trop grands")))
    };
    let matrix_bytes = (rows * cols)
        .checked_mul(dtype.bitsize() / 8)
        .filter(|&bytes| bytes > 0)
        .ok_or_else(|| {
            InferError::Dimension(format!("experts MTP {key} : dtype {dtype:?} non supporté"))
        })?;
    if matrix_bytes > stride {
        return Err(InferError::Dimension(format!(
            "experts MTP {key} : matrice [{rows}, {cols}] de {matrix_bytes} octets, stride de {stride}"
        )));
    }
    let (packed, scales, biases) = metal.weight_buffers_affine_with(
        total(packed_len)?,
        total(affine_len)?,
        |packed, scales, biases| {
            packed
                .par_chunks_mut(packed_len)
                .zip(scales.par_chunks_mut(affine_len))
                .zip(biases.par_chunks_mut(affine_len))
                .enumerate()
                .try_for_each(|(expert, ((packed, scales), biases))| -> Result<()> {
                    let bytes = sidecar.read(offset + expert * stride, matrix_bytes)?;
                    let data = bytes_to_dense_f32(&bytes, dtype, key)?;
                    let mut scales_f32 = vec![0.0_f32; affine_len];
                    let mut biases_f32 = vec![0.0_f32; affine_len];
                    crate::quantization::quantize_affine_into(
                        &data,
                        cols,
                        MTP_QUANT_GROUP,
                        4,
                        packed,
                        &mut scales_f32,
                        &mut biases_f32,
                    )?;
                    for (out, value) in scales.iter_mut().zip(&scales_f32) {
                        *out = crate::quantization::bf16_bits(*value);
                    }
                    for (out, value) in biases.iter_mut().zip(&biases_f32) {
                        *out = crate::quantization::bf16_bits(*value);
                    }
                    Ok(())
                })
        },
    )?;
    (0..experts)
        .map(|expert| {
            AffineQuantizedTensor::new_metal_shared_bf16(
                &[rows, packed_cols],
                packed.clone(),
                expert * packed_len,
                packed_len,
                &[rows, groups],
                scales.clone(),
                expert * affine_len,
                affine_len,
                &[rows, groups],
                biases.clone(),
                expert * affine_len,
                affine_len,
                MTP_QUANT_GROUP,
                4,
            )
            .map(LinearWeight::AffineQuantized)
        })
        .collect()
}

/// Sidecar dont les experts fusionnés restent sur disque.
///
/// `bytes` a la taille du fichier pour que `SafeTensors` valide le header,
/// mais seuls le header et les petits tenseurs y sont lus. Les pages des
/// experts (1,6 Go en bf16 sur le 35B-A3B) ne sont jamais touchées : un
/// `vec![0; n]` de cette taille vient de `calloc`, en pages zéro paresseuses
/// hors de l'empreinte physique. Les experts sont lus expert par expert.
///
/// NOTE: `read_metadata` seul suffirait à trouver les offsets, mais les
/// chemins dense et MoE consomment des `TensorView` de `SafeTensors`, qui
/// exige un buffer de la taille exacte du fichier. Le coût n'est que virtuel.
struct LazySidecar {
    file: std::fs::File,
    bytes: Vec<u8>,
    /// Tenseur réel → décalage de ses données dans le fichier.
    offsets: HashMap<String, usize>,
}

impl LazySidecar {
    fn open(path: &Path) -> Result<Self> {
        let io =
            |source: std::io::Error| InferError::Config(format!("lecture sidecar MTP: {source}"));
        let file = std::fs::File::open(path).map_err(io)?;
        let len = usize::try_from(file.metadata().map_err(io)?.len())
            .map_err(|_| InferError::Config("sidecar MTP trop grand".to_string()))?;
        let mut bytes = vec![0_u8; len];
        let mut len_bytes = [0_u8; 8];
        pread_exact(&file, &mut len_bytes, 0).map_err(io)?;
        let data_start = usize::try_from(u64::from_le_bytes(len_bytes))
            .ok()
            .and_then(|header| header.checked_add(8))
            .filter(|&start| start <= len)
            .ok_or_else(|| InferError::Config("header sidecar MTP invalide".to_string()))?;
        pread_exact(&file, &mut bytes[..data_start], 0).map_err(io)?;
        let (_, metadata) = SafeTensors::read_metadata(&bytes)
            .map_err(|source| InferError::Config(format!("parse sidecar MTP: {source}")))?;
        let mut offsets = HashMap::new();
        for (name, info) in metadata.tensors() {
            let (start, end) = info.data_offsets;
            // `read_metadata` a validé `end` contre la taille du fichier.
            let start = data_start + start;
            let end = data_start + end;
            if !matches!(
                normalize_mtp_sidecar_key(&name).as_str(),
                MTP_MOE_MARKER | MTP_MOE_DOWN
            ) {
                pread_exact(&file, &mut bytes[start..end], start as u64).map_err(io)?;
            }
            offsets.insert(name, start);
        }
        Ok(Self {
            file,
            bytes,
            offsets,
        })
    }

    /// Décalage fichier des données du tenseur de clé normalisée `norm_key`.
    fn data_offset(&self, resolve: &HashMap<String, String>, norm_key: &str) -> Result<usize> {
        resolve
            .get(norm_key)
            .and_then(|actual| self.offsets.get(actual))
            .copied()
            .ok_or_else(|| InferError::MissingWeight(norm_key.to_string()))
    }

    fn read(&self, offset: usize, len: usize) -> Result<Vec<u8>> {
        let mut out = vec![0_u8; len];
        pread_exact(&self.file, &mut out, offset as u64)
            .map_err(|source| InferError::Config(format!("lecture experts MTP: {source}")))?;
        Ok(out)
    }
}

/// Concatène ligne à ligne deux matrices de même hauteur : `[n, a] ++ [n, b] -> [n, a+b]`.
pub(super) fn concat_rows(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let (rows, a) = left.as_matrix()?;
    let (rows_right, b) = right.as_matrix()?;
    if rows != rows_right {
        return Err(InferError::Dimension(format!(
            "concat_rows: {rows} lignes contre {rows_right}"
        )));
    }
    let mut out = Vec::with_capacity(rows * (a + b));
    for row in 0..rows {
        out.extend_from_slice(left.row_slice(row)?);
        out.extend_from_slice(right.row_slice(row)?);
    }
    Tensor::from_vec(vec![rows, a + b], out)
}

pub(super) fn concat_row_pair(left: &Tensor, right: &Tensor) -> Result<Tensor> {
    let left = left.as_row()?;
    let right = right.as_row()?;
    let mut out = Vec::with_capacity(left.len() + right.len());
    out.extend_from_slice(left);
    out.extend_from_slice(right);
    Tensor::row(out)
}

fn normalize_mtp_sidecar_key(key: &str) -> String {
    if let Some(rest) = key.strip_prefix("language_model.mtp.") {
        format!("mtp.{rest}")
    } else if key.starts_with("mtp.") {
        key.to_string()
    } else if is_bare_mtp_sidecar_key(key) {
        format!("mtp.{key}")
    } else {
        key.to_string()
    }
}

fn is_bare_mtp_sidecar_key(key: &str) -> bool {
    key.starts_with("fc.")
        || key.starts_with("layers.")
        || key.starts_with("pre_fc_norm_")
        || key.starts_with("norm.")
}

/// Résout une vue tensor par clé normalisée.
fn st_view<'a>(
    st: &'a SafeTensors,
    resolve: &HashMap<String, String>,
    norm_key: &str,
) -> Result<TensorView<'a>> {
    let actual = resolve
        .get(norm_key)
        .ok_or_else(|| InferError::MissingWeight(norm_key.to_string()))?;
    st.tensor(actual)
        .map_err(|source| InferError::Config(format!("MTP tensor {norm_key}: {source}")))
}

/// Charge une norme `[dim]` selon la convention du sidecar.
fn load_mtp_norm(
    st: &SafeTensors,
    resolve: &HashMap<String, String>,
    key: &str,
    convention: MtpNormConvention,
) -> Result<Tensor> {
    let view = st_view(st, resolve, key)?;
    let data = bytes_to_dense_f32(view.data(), view.dtype(), key)?;
    let tensor = Tensor::from_vec(view.shape().to_vec(), data)?;
    match convention {
        MtpNormConvention::MeanHeuristic => {
            if tensor.rank() == 1 && !tensor.is_empty() {
                let mean = tensor.data().iter().sum::<f32>() / tensor.len() as f32;
                if mean < 0.5 {
                    return Ok(tensor.map(|value| value + 1.0));
                }
            }
            Ok(tensor)
        }
        MtpNormConvention::UnitOffset => {
            if tensor.rank() != 1 || tensor.is_empty() {
                return Err(InferError::Dimension(format!(
                    "norme MTP {key} : forme {:?}, rang 1 attendu",
                    tensor.shape()
                )));
            }
            if tensor.data().iter().any(|value| !value.is_finite()) {
                return Err(InferError::Config(format!(
                    "norme MTP {key} : valeur non finie"
                )));
            }
            Ok(tensor.map(|value| value + 1.0))
        }
    }
}

/// Charge un Linear MTP : quantifié 4-bit gs64 si `.scales` présent, sinon dense bf16.
fn load_mtp_linear(
    st: &SafeTensors,
    resolve: &HashMap<String, String>,
    prefix: &str,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let weight = st_view(st, resolve, &format!("{prefix}.weight"))?;
    let scales_key = format!("{prefix}.scales");
    if resolve.contains_key(&scales_key) {
        // Quantifié affine 4-bit (.weight u32 packé, .scales/.biases bf16).
        let packed_shape = weight.shape().to_vec();
        let scales_view = st_view(st, resolve, &scales_key)?;
        let biases_key = format!("{prefix}.biases");
        let biases_view = st_view(st, resolve, &biases_key)?;
        let (_, group_size) = infer_mtp_affine_layout(
            &packed_shape,
            scales_view.shape(),
            Some(4),
            None,
            &scales_key,
        )?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(quant) = mtp_affine_from_native_bf16(
            &weight,
            &scales_view,
            &biases_view,
            group_size,
            4,
            _context,
            prefix,
        )? {
            return Linear::from_weight(LinearWeight::AffineQuantized(quant), None);
        }
        let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
        let scales = Tensor::from_vec(
            scales_view.shape().to_vec(),
            bytes_to_dense_f32(scales_view.data(), scales_view.dtype(), &scales_key)?,
        )?;
        let biases = Tensor::from_vec(
            biases_view.shape().to_vec(),
            bytes_to_dense_f32(biases_view.data(), biases_view.dtype(), &biases_key)?,
        )?;
        let quant =
            AffineQuantizedTensor::new(&packed_shape, packed, scales, biases, group_size, 4)?;
        Linear::from_weight(LinearWeight::AffineQuantized(quant), None)
    } else {
        // Dense bf16 (ex. mtp.fc).
        let data = bytes_to_dense_f32(weight.data(), weight.dtype(), prefix)?;
        let dense = Tensor::from_vec(weight.shape().to_vec(), data)?;
        Linear::new(dense, None)
    }
}

pub(super) fn load_mtp_draft_lm_head(
    path: impl AsRef<Path>,
    expected_in_dim: usize,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|source| InferError::Config(format!("lecture draft lm_head MTP: {source}")))?;
    let st = SafeTensors::deserialize(&bytes)
        .map_err(|source| InferError::Config(format!("parse draft lm_head MTP: {source}")))?;
    let resolve: HashMap<String, String> = st
        .names()
        .into_iter()
        .map(|name| (normalize_mtp_draft_lm_head_key(name), name.to_string()))
        .collect();
    let head = load_mtp_draft_linear(&st, &resolve, "lm_head", expected_in_dim, _context)?;
    let shape = head.weight().shape();
    if shape.len() != 2 || shape[1] != expected_in_dim {
        return Err(InferError::Dimension(format!(
            "draft lm_head MTP input={}, attendu {expected_in_dim}",
            shape.get(1).copied().unwrap_or(0)
        )));
    }
    Ok(head)
}

fn normalize_mtp_draft_lm_head_key(key: &str) -> String {
    for prefix in [
        "language_model._mtplx_draft_lm_head.",
        "model._mtplx_draft_lm_head.",
        "_mtplx_draft_lm_head.",
        "language_model.lm_head.",
        "model.lm_head.",
        "lm_head.",
    ] {
        if let Some(rest) = key.strip_prefix(prefix) {
            return format!("lm_head.{rest}");
        }
    }
    if matches!(key, "weight" | "scales" | "biases") {
        format!("lm_head.{key}")
    } else {
        key.to_string()
    }
}

fn load_mtp_draft_linear(
    st: &SafeTensors,
    resolve: &HashMap<String, String>,
    prefix: &str,
    expected_in_dim: usize,
    _context: &MtpLoadContext<'_>,
) -> Result<Linear> {
    let weight = st_view(st, resolve, &format!("{prefix}.weight"))?;
    let scales_key = format!("{prefix}.scales");
    if resolve.contains_key(&scales_key) {
        let packed_shape = weight.shape().to_vec();
        let scales_view = st_view(st, resolve, &scales_key)?;
        let biases_key = format!("{prefix}.biases");
        let biases_view = st_view(st, resolve, &biases_key)?;
        let (bits, group_size) = infer_mtp_affine_layout(
            &packed_shape,
            scales_view.shape(),
            None,
            Some(expected_in_dim),
            &scales_key,
        )?;
        #[cfg(all(target_os = "macos", feature = "metal"))]
        if let Some(quant) = mtp_affine_from_native_bf16(
            &weight,
            &scales_view,
            &biases_view,
            group_size,
            bits,
            _context,
            prefix,
        )? {
            return Linear::from_weight(LinearWeight::AffineQuantized(quant), None);
        }
        let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
        let scales = Tensor::from_vec(
            scales_view.shape().to_vec(),
            bytes_to_dense_f32(scales_view.data(), scales_view.dtype(), &scales_key)?,
        )?;
        let biases = Tensor::from_vec(
            biases_view.shape().to_vec(),
            bytes_to_dense_f32(biases_view.data(), biases_view.dtype(), &biases_key)?,
        )?;
        let quant =
            AffineQuantizedTensor::new(&packed_shape, packed, scales, biases, group_size, bits)?;
        Linear::from_weight(LinearWeight::AffineQuantized(quant), None)
    } else {
        let data = bytes_to_dense_f32(weight.data(), weight.dtype(), prefix)?;
        let dense = Tensor::from_vec(weight.shape().to_vec(), data)?;
        Linear::new(dense, None)
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
fn mtp_affine_from_native_bf16(
    weight: &TensorView<'_>,
    scales: &TensorView<'_>,
    biases: &TensorView<'_>,
    group_size: usize,
    bits: usize,
    context: &MtpLoadContext<'_>,
    prefix: &str,
) -> Result<Option<AffineQuantizedTensor>> {
    if !crate::runtime_flags::single_copy_weights_enabled()
        || scales.dtype() != Dtype::BF16
        || biases.dtype() != Dtype::BF16
    {
        return Ok(None);
    }
    let Some(metal) = context.metal else {
        return Ok(None);
    };
    let packed = bytes_to_u32(weight.data(), &format!("{prefix}.weight"))?;
    let packed_len = packed.len();
    let scales_len = scales.data().len() / std::mem::size_of::<u16>();
    let biases_len = biases.data().len() / std::mem::size_of::<u16>();
    Ok(Some(AffineQuantizedTensor::new_metal_shared_bf16(
        weight.shape(),
        metal.weight_buffer_from_u32(&packed)?,
        0,
        packed_len,
        scales.shape(),
        metal.weight_buffer_from_bf16_bytes(scales.data())?,
        0,
        scales_len,
        biases.shape(),
        metal.weight_buffer_from_bf16_bytes(biases.data())?,
        0,
        biases_len,
        group_size,
        bits,
    )?))
}

fn infer_mtp_affine_layout(
    packed_shape: &[usize],
    scales_shape: &[usize],
    fixed_bits: Option<usize>,
    expected_cols: Option<usize>,
    key: &str,
) -> Result<(usize, usize)> {
    let &[rows, packed_cols] = packed_shape else {
        return Err(InferError::Dimension(format!(
            "MTP {key} poids quantifié attendu rang 2, reçu {packed_shape:?}"
        )));
    };
    let &[scale_rows, groups] = scales_shape else {
        return Err(InferError::Dimension(format!(
            "MTP {key} scales attendues rang 2, reçu {scales_shape:?}"
        )));
    };
    if scale_rows != rows {
        return Err(InferError::Dimension(format!(
            "MTP {key} scales rows={scale_rows}, poids rows={rows}"
        )));
    }
    if groups == 0 {
        return Err(InferError::Shape(format!("MTP {key} scales sans groupe")));
    }
    let packed_bits = packed_cols
        .checked_mul(32)
        .ok_or_else(|| InferError::Shape(format!("MTP {key} poids trop large")))?;
    let bits = match (fixed_bits, expected_cols) {
        (Some(bits), _) => bits,
        (None, Some(cols)) if cols > 0 && packed_bits % cols == 0 => packed_bits / cols,
        (None, Some(cols)) => {
            return Err(InferError::Dimension(format!(
                "MTP {key} packed_bits={packed_bits} non divisible par input={cols}"
            )))
        }
        (None, None) => {
            return Err(InferError::Config(format!(
                "MTP {key} bits ou input attendu requis"
            )))
        }
    };
    if bits == 0 || bits > 16 {
        return Err(InferError::Config(format!(
            "MTP {key} bits inférés invalides: {bits}"
        )));
    }
    let cols = match expected_cols {
        Some(cols) => cols,
        None if packed_bits % bits == 0 => packed_bits / bits,
        None => {
            return Err(InferError::Shape(format!(
                "MTP {key} packed_cols={packed_cols} incompatible avec bits={bits}"
            )))
        }
    };
    if cols == 0 || cols % groups != 0 {
        return Err(InferError::Dimension(format!(
            "MTP {key} cols={cols} incompatibles avec groups={groups}"
        )));
    }
    Ok((bits, cols / groups))
}

#[cfg(test)]
mod tests {
    use super::{concat_rows, infer_mtp_affine_layout, normalize_mtp_sidecar_key};
    use crate::tensor::Tensor;

    #[test]
    fn concat_rows_interleaves_each_row() {
        let left = Tensor::from_vec(vec![2, 2], vec![1.0, 2.0, 5.0, 6.0])
            .expect("invariant: forme valide");
        let right = Tensor::from_vec(vec![2, 1], vec![3.0, 7.0]).expect("invariant: forme valide");
        let out = concat_rows(&left, &right).expect("invariant: hauteurs égales");
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 5.0, 6.0, 7.0]);
        let short = Tensor::from_vec(vec![1, 1], vec![0.0]).expect("invariant: forme valide");
        assert!(concat_rows(&left, &short).is_err());
    }

    #[test]
    fn normalizes_bare_quantized_fc_sidecar_keys() {
        // Sidecar mlx-community/*-MTP-4bit : `fc` quantifié (weight + scales + biases).
        for key in ["fc.weight", "fc.scales", "fc.biases"] {
            assert_eq!(normalize_mtp_sidecar_key(key), format!("mtp.{key}"));
        }
        assert_eq!(normalize_mtp_sidecar_key("mtp.fc.weight"), "mtp.fc.weight");
        assert_eq!(
            normalize_mtp_sidecar_key("lm_head.weight"),
            "lm_head.weight"
        );
    }

    #[test]
    fn infer_mtp_affine_group_size_accepts_gs32_and_gs64() {
        let (_, gs32) = infer_mtp_affine_layout(&[12_288, 640], &[12_288, 160], Some(4), None, "q")
            .expect("invariant: gs32 MTPLX valide");
        let (_, gs64) = infer_mtp_affine_layout(&[12_288, 640], &[12_288, 80], Some(4), None, "q")
            .expect("invariant: gs64 OptiQ valide");
        assert_eq!(gs32, 32);
        assert_eq!(gs64, 64);
    }

    #[test]
    fn infer_mtp_affine_layout_accepts_q3_draft_head() {
        let (bits, group_size) =
            infer_mtp_affine_layout(&[152_064, 480], &[152_064, 80], None, Some(5120), "head")
                .expect("invariant: q3 gs64 draft head valide");
        assert_eq!(bits, 3);
        assert_eq!(group_size, 64);
    }

    mod moe_head {
        use super::super::{MtpHead, MtpLoadContext, MtpMoeQuant};
        use crate::decoder::CausalDecoderConfig;
        use crate::{FeedForward, InferError, LinearWeight, Tensor};
        use safetensors::{tensor::TensorView, Dtype, SafeTensors};

        const H: usize = 64;
        const I: usize = 64;
        const E: usize = 4;

        type Raw = (Vec<usize>, Vec<u8>);

        /// Tenseur f32 sérialisable : valeurs `f(index)`.
        fn tensor(shape: &[usize], f: impl Fn(usize) -> f32) -> Raw {
            let len = shape.iter().product::<usize>();
            let bytes = (0..len).flat_map(|i| f(i).to_le_bytes()).collect();
            (shape.to_vec(), bytes)
        }

        fn ramp(i: usize) -> f32 {
            (i % H) as f32 * 1e-3
        }

        /// Tête MoE synthétique au format transformers : gate de l'expert `e`
        /// = `1 + e`, up = `-(1 + e)`, plus une rampe par colonne pour que les
        /// groupes restent quantifiables.
        fn moe_tensors() -> Vec<(String, Raw)> {
            let mut out: Vec<(String, Raw)> = vec![
                ("mtp.fc.weight".into(), tensor(&[H, 2 * H], ramp)),
                (
                    "mtp.pre_fc_norm_embedding.weight".into(),
                    tensor(&[H], |_| 0.25),
                ),
                (
                    "mtp.pre_fc_norm_hidden.weight".into(),
                    tensor(&[H], |_| 0.5),
                ),
                ("mtp.norm.weight".into(), tensor(&[H], |_| 1.875)),
                (
                    "mtp.layers.0.input_layernorm.weight".into(),
                    tensor(&[H], |_| 0.0),
                ),
                (
                    "mtp.layers.0.post_attention_layernorm.weight".into(),
                    tensor(&[H], |_| 0.75),
                ),
                (
                    "mtp.layers.0.self_attn.q_norm.weight".into(),
                    tensor(&[16], |_| 0.625),
                ),
                (
                    "mtp.layers.0.self_attn.k_norm.weight".into(),
                    tensor(&[16], |_| 0.625),
                ),
                ("mtp.layers.0.mlp.gate.weight".into(), tensor(&[E, H], ramp)),
                (
                    "mtp.layers.0.mlp.shared_expert_gate.weight".into(),
                    tensor(&[1, H], ramp),
                ),
                (
                    "mtp.layers.0.mlp.experts.gate_up_proj".into(),
                    tensor(&[E, 2 * I, H], |i| {
                        let expert = i / (2 * I * H);
                        let row = (i / H) % (2 * I);
                        let sign = if row < I { 1.0 } else { -1.0 };
                        sign * (1 + expert) as f32 + ramp(i)
                    }),
                ),
                (
                    "mtp.layers.0.mlp.experts.down_proj".into(),
                    tensor(&[E, H, I], |i| (i / (H * I)) as f32 + ramp(i)),
                ),
                (
                    "mtp.layers.0.mlp.shared_expert.down_proj.weight".into(),
                    tensor(&[H, I], ramp),
                ),
            ];
            // 4 têtes Q à porte (4 × 16 × 2 = 128 sorties), 4 têtes KV de 16.
            out.push((
                "mtp.layers.0.self_attn.q_proj.weight".into(),
                tensor(&[2 * H, H], ramp),
            ));
            for name in ["k_proj", "v_proj"] {
                out.push((
                    format!("mtp.layers.0.self_attn.{name}.weight"),
                    tensor(&[H, H], ramp),
                ));
            }
            out.push((
                "mtp.layers.0.self_attn.o_proj.weight".into(),
                tensor(&[H, H], ramp),
            ));
            for name in ["gate_proj", "up_proj"] {
                out.push((
                    format!("mtp.layers.0.mlp.shared_expert.{name}.weight"),
                    tensor(&[I, H], ramp),
                ));
            }
            out
        }

        fn load(tensors: &[(String, Raw)], quant: MtpMoeQuant) -> crate::Result<MtpHead> {
            load_with_head_dim(tensors, quant, Some(16))
        }

        fn load_with_head_dim(
            tensors: &[(String, Raw)],
            quant: MtpMoeQuant,
            head_dim: Option<usize>,
        ) -> crate::Result<MtpHead> {
            load_in(tensors, quant, head_dim, &MtpLoadContext::cpu())
        }

        /// Tenseur typé d'un sidecar de test.
        type Typed = (String, Dtype, Vec<usize>, Vec<u8>);

        fn f32_typed(tensors: &[(String, Raw)]) -> Vec<Typed> {
            tensors
                .iter()
                .map(|(name, (shape, bytes))| {
                    (name.clone(), Dtype::F32, shape.clone(), bytes.clone())
                })
                .collect()
        }

        fn write_sidecar(tensors: &[Typed]) -> tempfile::NamedTempFile {
            let views = tensors
                .iter()
                .map(|(name, dtype, shape, bytes)| {
                    TensorView::new(*dtype, shape.clone(), bytes).map(|view| (name.clone(), view))
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("invariant: vues safetensors valides");
            let buffer = safetensors::serialize(views, None).expect("invariant: sérialisation");
            let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
            std::fs::write(tmp.path(), buffer).expect("invariant: écriture sidecar");
            tmp
        }

        fn moe_config(head_dim: Option<usize>, experts: usize) -> CausalDecoderConfig {
            CausalDecoderConfig {
                attn_output_gate: true,
                head_dim,
                num_attention_heads: 4,
                num_key_value_heads: 4,
                num_experts: Some(experts),
                num_experts_per_tok: 2,
                ..CausalDecoderConfig::default()
            }
        }

        fn load_in(
            tensors: &[(String, Raw)],
            quant: MtpMoeQuant,
            head_dim: Option<usize>,
            context: &MtpLoadContext<'_>,
        ) -> crate::Result<MtpHead> {
            let tmp = write_sidecar(&f32_typed(tensors));
            MtpHead::from_sidecar_with(tmp.path(), &moe_config(head_dim, E), H, context, quant)
        }

        fn first(weight: &LinearWeight) -> Option<f32> {
            match weight {
                LinearWeight::Dense(tensor) => tensor.data().first().copied(),
                #[cfg(all(target_os = "macos", feature = "metal"))]
                LinearWeight::DenseBf16(tensor) => tensor.row(0).ok()?.first().copied(),
                LinearWeight::AffineQuantized(_) => None,
            }
        }

        #[test]
        fn fused_experts_split_gate_first_and_norms_get_unit_offset() {
            let head = load(&moe_tensors(), MtpMoeQuant::Dense).expect("invariant: tête MoE");
            let FeedForward::Moe(moe) = &head.layer.mlp else {
                panic!("MLP MoE attendue");
            };
            let (_, experts, top_k) = moe.routed_parts_for_test();
            assert_eq!((experts.len(), top_k), (E, 2));
            for (expert, mlp) in experts.iter().enumerate() {
                let (gate, up, down) = mlp.projections();
                let base = (1 + expert) as f32;
                assert_eq!(first(gate.weight()), Some(base), "gate = première moitié");
                assert_eq!(first(up.weight()), Some(-base), "up = seconde moitié");
                assert_eq!(down.weight().shape(), &[H, I]);
                assert_eq!(first(down.weight()), Some(expert as f32));
            }
            // Convention officielle : `1 + weight` partout, même au-delà de 0,5.
            assert_eq!(head.pre_fc_norm_embedding.data()[0], 1.25);
            assert_eq!(head.pre_fc_norm_hidden.data()[0], 1.5);
            assert_eq!(head.layer.input_norm.data()[0], 1.0);
            assert_eq!(head.layer.post_attention_norm.data()[0], 1.75);
            assert_eq!(head.norm.data()[0], 2.875);
            let q_norm = head.layer.attention.q_norm.as_ref().map(|n| n.data()[0]);
            assert_eq!(q_norm, Some(1.625));
        }

        #[test]
        fn head_dim_is_inferred_from_q_proj_without_override() {
            // `CausalDecoderConfig::head_dim` facultatif : déduit de q_proj
            // (128 sorties, 4 têtes à porte → 16), comme l'attention du tronc.
            let head = load_with_head_dim(&moe_tensors(), MtpMoeQuant::Dense, None)
                .expect("invariant: head_dim déductible");
            let q_norm = head.layer.attention.q_norm.as_ref().map(Tensor::len);
            assert_eq!(q_norm, Some(16));
        }

        #[test]
        fn affine_moe_head_is_fully_quantized() {
            let head = load(&moe_tensors(), MtpMoeQuant::Affine).expect("invariant: tête MoE");
            let FeedForward::Moe(moe) = &head.layer.mlp else {
                panic!("MLP MoE attendue");
            };
            let (router, experts, _) = moe.routed_parts_for_test();
            let affine = |weight: &LinearWeight| matches!(weight, LinearWeight::AffineQuantized(_));
            assert!(affine(router.weight()) && affine(head.fc.weight()));
            assert!(affine(head.layer.attention.q_proj.weight()));
            for mlp in experts {
                let (gate, up, down) = mlp.projections();
                assert!(affine(gate.weight()) && affine(up.weight()) && affine(down.weight()));
            }
        }

        /// Chemin Metal de la tête : quantification directe des experts,
        /// single-copy des linéaires par `with_mtp_sidecar` et forward CPU
        /// après conversion. Chaque test suit `RETI_RUST_SINGLE_COPY_WEIGHTS`
        /// (mémorisé par processus) : coupé, la tête doit rester sur le CPU.
        #[cfg(all(target_os = "macos", feature = "metal"))]
        mod metal {
            use super::*;
            use crate::decoder::mtp::{MTP_MOE_DOWN, MTP_MOE_MARKER, MTP_QUANT_GROUP};
            use crate::decoder::{CausalDecoder, LayerKvCache};
            use crate::runtime_flags::single_copy_weights_enabled;
            use crate::{AffineQuantizedTensor, ForwardRuntime, Linear};
            use std::collections::HashMap;

            const H2: usize = 128;
            const I2: usize = 192;
            const E2: usize = 3;
            const VOCAB: usize = 8;

            fn executor() -> crate::Result<Option<crate::MetalExecutor>> {
                match crate::MetalExecutor::new() {
                    Ok(executor) => Ok(Some(executor)),
                    Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
                    Err(error) => Err(error),
                }
            }

            /// Arrondi bf16 au plus proche, égalités au pair, par comparaison
            /// des deux voisins en f64 : indépendant de `bf16_bits`.
            fn bf16_oracle(value: f32) -> f32 {
                let bits = value.to_bits();
                let low = f32::from_bits(bits & 0xffff_0000);
                let high = f32::from_bits((bits & 0xffff_0000).wrapping_add(0x1_0000));
                let to_low = (f64::from(value) - f64::from(low)).abs();
                let to_high = (f64::from(high) - f64::from(value)).abs();
                if to_low < to_high || (to_low == to_high && (bits >> 16) & 1 == 0) {
                    low
                } else {
                    high
                }
            }

            fn bf16_round_trip(value: f32) -> f32 {
                f32::from_bits(u32::from(crate::quantization::bf16_bits(value)) << 16)
            }

            #[test]
            fn bf16_bits_matches_independent_oracle_on_ties() {
                // Égalités exactes : au pair, vers le bas puis vers le haut.
                for (input, expected) in [
                    (0x3F80_8000_u32, 0x3F80_u16),
                    (0x3F81_8000, 0x3F82),
                    (0xBF80_8000, 0xBF80),
                    (0xBF81_8000, 0xBF82),
                    (0x3F80_8001, 0x3F81),
                    (0x3F80_7FFF, 0x3F80),
                    (0x0000_8000, 0x0000),
                    (0x0001_8000, 0x0002),
                ] {
                    let value = f32::from_bits(input);
                    assert_eq!(
                        crate::quantization::bf16_bits(value),
                        expected,
                        "{input:#x}"
                    );
                    assert_eq!(bf16_oracle(value).to_bits() >> 16, u32::from(expected));
                }
                // Balayage pseudo-aléatoire des valeurs finies.
                let mut state = 0x9E37_79B9_u32;
                for _ in 0..100_000 {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let value = f32::from_bits(state);
                    // Voisin haut infini : débordement, hors du domaine de l'oracle.
                    let high = f32::from_bits((state & 0xffff_0000).wrapping_add(0x1_0000));
                    if value.is_finite() && high.is_finite() {
                        assert_eq!(
                            bf16_round_trip(value).to_bits(),
                            bf16_oracle(value).to_bits(),
                            "{state:#x}"
                        );
                    }
                }
            }

            /// Valeur distincte par tenseur, expert, ligne et groupe de 64.
            fn distinct(seed: usize, expert: usize, row: usize, col: usize) -> f32 {
                let hash = (seed * 104_729 + expert * 7_919 + row * 131 + col * 17) % 1_009;
                let group = col / MTP_QUANT_GROUP;
                (hash as f32 / 1_009.0 - 0.5) * 0.05 * (1 + group) as f32 + 0.01 * expert as f32
            }

            /// Valeur bf16 stockée (troncature : c'est elle qui fait foi).
            fn stored(value: f32) -> f32 {
                f32::from_bits(value.to_bits() & 0xffff_0000)
            }

            fn bf16_bytes(len: usize, f: impl Fn(usize) -> f32) -> Vec<u8> {
                (0..len)
                    .flat_map(|i| ((f(i).to_bits() >> 16) as u16).to_le_bytes())
                    .collect()
            }

            fn bf16_matrix(name: &str, seed: usize, rows: usize, cols: usize) -> Typed {
                let bytes = bf16_bytes(rows * cols, |i| distinct(seed, 0, i / cols, i % cols));
                (name.to_string(), Dtype::BF16, vec![rows, cols], bytes)
            }

            fn bf16_vector(name: &str, len: usize, value: f32) -> Typed {
                (
                    name.to_string(),
                    Dtype::BF16,
                    vec![len],
                    bf16_bytes(len, |_| value),
                )
            }

            /// Tête MoE bf16 non carrée : hidden 128 (2 groupes), intermédiaire
            /// 192 (3 groupes), 3 experts, valeurs distinctes partout.
            fn bf16_moe_tensors() -> Vec<Typed> {
                let mut out = vec![
                    bf16_matrix("mtp.fc.weight", 10, H2, 2 * H2),
                    bf16_vector("mtp.pre_fc_norm_embedding.weight", H2, 0.25),
                    bf16_vector("mtp.pre_fc_norm_hidden.weight", H2, 0.5),
                    bf16_vector("mtp.norm.weight", H2, 0.875),
                    bf16_vector("mtp.layers.0.input_layernorm.weight", H2, 0.0),
                    bf16_vector("mtp.layers.0.post_attention_layernorm.weight", H2, 0.75),
                    bf16_vector("mtp.layers.0.self_attn.q_norm.weight", 16, 0.625),
                    bf16_vector("mtp.layers.0.self_attn.k_norm.weight", 16, 0.625),
                    bf16_matrix("mtp.layers.0.self_attn.q_proj.weight", 11, 128, H2),
                    bf16_matrix("mtp.layers.0.self_attn.k_proj.weight", 12, 64, H2),
                    bf16_matrix("mtp.layers.0.self_attn.v_proj.weight", 13, 64, H2),
                    bf16_matrix("mtp.layers.0.self_attn.o_proj.weight", 14, H2, 64),
                    bf16_matrix("mtp.layers.0.mlp.gate.weight", 15, E2, H2),
                    bf16_matrix("mtp.layers.0.mlp.shared_expert_gate.weight", 16, 1, H2),
                    bf16_matrix(
                        "mtp.layers.0.mlp.shared_expert.gate_proj.weight",
                        17,
                        I2,
                        H2,
                    ),
                    bf16_matrix("mtp.layers.0.mlp.shared_expert.up_proj.weight", 18, I2, H2),
                    bf16_matrix(
                        "mtp.layers.0.mlp.shared_expert.down_proj.weight",
                        19,
                        H2,
                        I2,
                    ),
                ];
                out.push((
                    MTP_MOE_MARKER.to_string(),
                    Dtype::BF16,
                    vec![E2, 2 * I2, H2],
                    bf16_bytes(E2 * 2 * I2 * H2, |i| {
                        distinct(1, i / (2 * I2 * H2), (i / H2) % (2 * I2), i % H2)
                    }),
                ));
                out.push((
                    MTP_MOE_DOWN.to_string(),
                    Dtype::BF16,
                    vec![E2, H2, I2],
                    bf16_bytes(E2 * H2 * I2, |i| {
                        distinct(2, i / (H2 * I2), (i / I2) % H2, i % I2)
                    }),
                ));
                out
            }

            /// Projection `projection` (gate, up, down) de l'expert `expert`,
            /// recalculée depuis la formule du fixture, sans le sidecar.
            fn expected_expert(projection: usize, expert: usize) -> crate::Result<Tensor> {
                // (lignes, colonnes, graine, décalage de ligne dans `gate_up`).
                let (rows, cols, seed, row_offset) = match projection {
                    0 => (I2, H2, 1, 0),
                    1 => (I2, H2, 1, I2),
                    _ => (H2, I2, 2, 0),
                };
                let data = (0..rows * cols)
                    .map(|i| stored(distinct(seed, expert, row_offset + i / cols, i % cols)))
                    .collect();
                Tensor::from_vec(vec![rows, cols], data)
            }

            fn routed(head: &MtpHead) -> &[crate::GatedMlp] {
                match &head.layer.mlp {
                    FeedForward::Moe(moe) => moe.routed_parts_for_test().1,
                    FeedForward::Dense(_) => &[],
                }
            }

            type View = (usize, usize);

            fn views(weight: &AffineQuantizedTensor) -> Option<[View; 3]> {
                let view = |v: Option<(&::metal::Buffer, usize)>| {
                    v.map(|(buffer, offset)| (buffer.contents() as usize, offset))
                };
                Some([
                    view(weight.packed_metal_view())?,
                    view(weight.scales_metal_view())?,
                    view(weight.biases_metal_view())?,
                ])
            }

            fn affine(linear: &Linear) -> &AffineQuantizedTensor {
                linear
                    .affine_quantized()
                    .expect("invariant: linéaire quantifiée attendue")
            }

            /// Scales ou minimums attendus : arrondis bf16 en single-copy,
            /// f32 du CPU sinon.
            fn expected_params(values: &[f32], single_copy: bool) -> Vec<f32> {
                values
                    .iter()
                    .map(|&v| if single_copy { bf16_oracle(v) } else { v })
                    .collect()
            }

            #[test]
            fn bf16_experts_quantize_into_contiguous_metal_stacks() -> crate::Result<()> {
                let Some(executor) = executor()? else {
                    return Ok(());
                };
                let single_copy = single_copy_weights_enabled();
                let tmp = write_sidecar(&bf16_moe_tensors());
                let config = moe_config(Some(16), E2);
                let load = |context: &MtpLoadContext<'_>| {
                    MtpHead::from_sidecar_with(
                        tmp.path(),
                        &config,
                        H2,
                        context,
                        MtpMoeQuant::Affine,
                    )
                };
                let cpu = load(&MtpLoadContext::cpu())?;
                let mut metal = load(&MtpLoadContext::metal(&executor))?;
                assert_eq!((routed(&cpu).len(), routed(&metal).len()), (E2, E2));
                let mut before = Vec::new();
                for (expert, (cpu_mlp, metal_mlp)) in
                    routed(&cpu).iter().zip(routed(&metal)).enumerate()
                {
                    let (cg, cu, cd) = cpu_mlp.projections();
                    let (mg, mu, md) = metal_mlp.projections();
                    for (projection, (cpu_linear, metal_linear)) in
                        [(cg, mg), (cu, mu), (cd, md)].into_iter().enumerate()
                    {
                        let reference = AffineQuantizedTensor::quantize(
                            &expected_expert(projection, expert)?,
                            MTP_QUANT_GROUP,
                            4,
                        )?;
                        let label = format!("expert {expert} projection {projection}");
                        // Lecture paresseuse : offsets gate/up/down et stride exacts.
                        let (cpu_w, metal_w) = (affine(cpu_linear), affine(metal_linear));
                        assert_eq!(cpu_w.packed_data(), reference.packed_data(), "CPU {label}");
                        assert_eq!(metal_w.packed_data(), reference.packed_data(), "{label}");
                        assert_eq!(*cpu_w.scales_f32(), *reference.scales_f32(), "{label}");
                        assert_eq!(
                            *metal_w.scales_f32(),
                            *expected_params(&reference.scales_f32(), single_copy),
                            "scales {label}"
                        );
                        assert_eq!(
                            *metal_w.biases_f32(),
                            *expected_params(&reference.biases_f32(), single_copy),
                            "minimums {label}"
                        );
                        assert!(views(cpu_w).is_none());
                        match views(metal_w) {
                            Some(view) => before.push((projection, expert, view, reference)),
                            None => assert!(!single_copy, "vue Metal attendue {label}"),
                        }
                    }
                }
                if !single_copy {
                    return Ok(());
                }
                // Stacks contigus : l'expert `e` suit le premier à e · stride
                // dans le même buffer, ce qu'exige `contiguous_metal_expert_stacks`.
                for (projection, expert, view, reference) in &before {
                    let (_, _, first, _) = before
                        .iter()
                        .find(|(p, e, _, _)| p == projection && *e == 0)
                        .expect("invariant: premier expert");
                    let strides = [
                        reference.packed_len() * 4,
                        reference.scales_len() * 2,
                        reference.biases_len() * 2,
                    ];
                    for ((got, start), stride) in view.iter().zip(first).zip(strides) {
                        assert_eq!(got.0, start.0, "même buffer");
                        assert_eq!(got.1, start.1 + expert * stride, "offset contigu");
                    }
                }
                // La passe single-copy du décodeur ne recopie rien.
                metal.move_packed_to_metal_shared(&executor)?;
                let after: Vec<_> = routed(&metal)
                    .iter()
                    .flat_map(|mlp| {
                        let (g, u, d) = mlp.projections();
                        [g, u, d].map(|linear| views(affine(linear)))
                    })
                    .collect();
                let mut expected: Vec<_> = before.iter().map(|b| (b.1, b.0, Some(b.2))).collect();
                expected.sort_by_key(|(e, p, _)| (*e, *p));
                assert_eq!(after, expected.into_iter().map(|e| e.2).collect::<Vec<_>>());
                Ok(())
            }

            /// Tronc minimal sans MLP, à la largeur de la tête ; l'attention du
            /// tronc ne tourne pas dans ces tests, d'où la config de la tête
            /// posée après construction.
            fn tiny_trunk(
                hidden: usize,
                experts: Option<usize>,
            ) -> crate::Result<Option<CausalDecoder>> {
                let matrix = |seed: usize, rows: usize, cols: usize| {
                    let data = (0..rows * cols)
                        .map(|i| distinct(seed, 0, i / cols, i % cols))
                        .collect();
                    Tensor::from_vec(vec![rows, cols], data)
                };
                let mut tensors = HashMap::new();
                tensors.insert(
                    "embed_tokens.weight".to_string(),
                    matrix(30, VOCAB, hidden)?,
                );
                tensors.insert("lm_head.weight".to_string(), matrix(31, VOCAB, hidden)?);
                tensors.insert(
                    "layers.0.input_layernorm.weight".to_string(),
                    Tensor::from_vec(vec![hidden], vec![1.0; hidden])?,
                );
                tensors.insert(
                    "norm.weight".to_string(),
                    Tensor::from_vec(vec![hidden], vec![1.0; hidden])?,
                );
                for (name, seed, rows, cols) in [
                    ("q_proj", 32, 64, hidden),
                    ("k_proj", 33, 64, hidden),
                    ("v_proj", 34, 64, hidden),
                    ("o_proj", 35, hidden, 64),
                ] {
                    tensors.insert(
                        format!("layers.0.self_attn.{name}.weight"),
                        matrix(seed, rows, cols)?,
                    );
                }
                let config = CausalDecoderConfig {
                    num_attention_heads: 4,
                    num_key_value_heads: 4,
                    head_dim: Some(16),
                    ..CausalDecoderConfig::default()
                };
                let mut model = CausalDecoder::from_tensors(tensors, config)?;
                model.config.attn_output_gate = true;
                model.config.num_experts = experts;
                model.config.num_experts_per_tok = 2;
                match model.with_metal_runtime() {
                    Ok(model) => Ok(Some(model)),
                    Err(InferError::Metal(message)) if message.contains("aucun device") => Ok(None),
                    Err(error) => Err(error),
                }
            }

            /// Linéaires quantifiées de la tête, dans un ordre fixe.
            fn quantized_linears(head: &MtpHead) -> Vec<&Linear> {
                let attention = &head.layer.attention;
                let mut out = vec![&head.fc, &attention.q_proj, &attention.k_proj];
                out.extend(attention.v_proj.as_ref());
                out.push(&attention.o_proj);
                match &head.layer.mlp {
                    FeedForward::Moe(moe) => {
                        let (router, _, _, shared, shared_gate) = moe
                            .shared_metal_parts()
                            .expect("invariant: MoE à expert partagé");
                        let (g, u, d) = shared.projections();
                        out.extend([router, g, u, d, shared_gate]);
                    }
                    FeedForward::Dense(mlp) => {
                        let (g, u, d) = mlp.projections();
                        out.extend([g, u, d]);
                    }
                }
                out.retain(|linear| linear.affine_quantized().is_some());
                out
            }

            fn cpu_forward(model: &CausalDecoder, head: &MtpHead) -> crate::Result<Vec<f32>> {
                let hidden = model.final_norm.data().len();
                let mut cache = LayerKvCache::default();
                let mut out = Vec::new();
                for (position, token) in [3_usize, 5].into_iter().enumerate() {
                    let row = Tensor::row(
                        (0..hidden)
                            .map(|c| distinct(40 + position, 0, 0, c))
                            .collect(),
                    )?;
                    let post = model.mtp_head_forward(
                        head,
                        &row,
                        token,
                        position,
                        &mut cache,
                        ForwardRuntime::cpu(),
                    )?;
                    out.extend_from_slice(post.data());
                }
                Ok(out)
            }

            /// Tête MoE par `with_mtp_sidecar` : fc, attention, routeur et expert
            /// partagé passent en single-copy (codes inchangés, paramètres bf16),
            /// et le forward CPU tourne sur les buffers partagés.
            #[test]
            fn with_mtp_sidecar_moves_moe_linears_and_cpu_forward_runs() -> crate::Result<()> {
                let Some(model) = tiny_trunk(H, Some(E))? else {
                    return Ok(());
                };
                let single_copy = single_copy_weights_enabled();
                let tmp = write_sidecar(&f32_typed(&moe_tensors()));
                let cpu_head = MtpHead::from_sidecar_with(
                    tmp.path(),
                    &model.config,
                    H,
                    &MtpLoadContext::cpu(),
                    MtpMoeQuant::Affine,
                )?;
                let model = model.with_mtp_sidecar(tmp.path())?;
                let head = model.mtp.as_ref().expect("invariant: tête chargée");
                let (cpu_linears, linears) =
                    (quantized_linears(&cpu_head), quantized_linears(head));
                assert_eq!((cpu_linears.len(), linears.len()), (10, 10));
                for (index, (cpu_linear, linear)) in cpu_linears.iter().zip(&linears).enumerate() {
                    let (cpu_w, w) = (affine(cpu_linear), affine(linear));
                    assert_eq!(w.packed_data(), cpu_w.packed_data(), "linéaire {index}");
                    assert_eq!(
                        *w.scales_f32(),
                        *expected_params(&cpu_w.scales_f32(), single_copy),
                        "linéaire {index}"
                    );
                    assert_eq!(views(w).is_some(), single_copy, "linéaire {index}");
                }
                let got = cpu_forward(&model, head)?;
                let reference = cpu_forward(&model, &cpu_head)?;
                let scale = reference.iter().fold(0.0_f32, |m, v| m.max(v.abs()));
                assert!(got.iter().all(|v| v.is_finite()) && scale > 0.0);
                let diff = got
                    .iter()
                    .zip(&reference)
                    .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
                // Seul écart : scales et minimums arrondis en bf16 (2^-8 relatif).
                assert!(
                    diff <= 2e-2 * scale,
                    "écart {diff} pour une échelle {scale}"
                );
                Ok(())
            }

            /// Linéaire OptiQ u4 gs64 sérialisée comme le sidecar du 27B :
            /// codes `u32`, scales et minimums bf16.
            fn optiq_linear(prefix: &str, seed: usize, rows: usize, cols: usize) -> Vec<Typed> {
                let data = (0..rows * cols)
                    .map(|i| distinct(seed, 0, i / cols, i % cols))
                    .collect();
                let dense =
                    Tensor::from_vec(vec![rows, cols], data).expect("invariant: matrice dense");
                let quant = AffineQuantizedTensor::quantize(&dense, MTP_QUANT_GROUP, 4)
                    .expect("invariant: quantification");
                let groups = cols / MTP_QUANT_GROUP;
                let params = |values: &[f32]| {
                    values
                        .iter()
                        .flat_map(|&v| crate::quantization::bf16_bits(v).to_le_bytes())
                        .collect::<Vec<u8>>()
                };
                vec![
                    (
                        format!("{prefix}.weight"),
                        Dtype::U32,
                        vec![rows, cols / 8],
                        quant
                            .packed_data()
                            .iter()
                            .flat_map(|v| v.to_le_bytes())
                            .collect(),
                    ),
                    (
                        format!("{prefix}.scales"),
                        Dtype::BF16,
                        vec![rows, groups],
                        params(&quant.scales_f32()),
                    ),
                    (
                        format!("{prefix}.biases"),
                        Dtype::BF16,
                        vec![rows, groups],
                        params(&quant.biases_f32()),
                    ),
                ]
            }

            /// Tête dense du 27B par `with_mtp_sidecar` : linéaires OptiQ en vues
            /// Metal natives, fc bf16 dense, forward CPU identique au bit près à
            /// la même tête chargée sur le CPU.
            #[test]
            fn with_mtp_sidecar_dense_27b_head_matches_cpu_load() -> crate::Result<()> {
                let Some(model) = tiny_trunk(H, None)? else {
                    return Ok(());
                };
                let single_copy = single_copy_weights_enabled();
                let mut tensors = vec![
                    bf16_matrix("mtp.fc.weight", 20, H, 2 * H),
                    bf16_vector("mtp.pre_fc_norm_embedding.weight", H, 1.0),
                    bf16_vector("mtp.pre_fc_norm_hidden.weight", H, 1.0),
                    bf16_vector("mtp.norm.weight", H, 1.0),
                    bf16_vector("mtp.layers.0.input_layernorm.weight", H, 1.0),
                    bf16_vector("mtp.layers.0.post_attention_layernorm.weight", H, 1.0),
                    bf16_vector("mtp.layers.0.self_attn.q_norm.weight", 16, 1.0),
                    bf16_vector("mtp.layers.0.self_attn.k_norm.weight", 16, 1.0),
                ];
                for (name, seed, rows, cols) in [
                    ("self_attn.q_proj", 21, 2 * H, H),
                    ("self_attn.k_proj", 22, H, H),
                    ("self_attn.v_proj", 23, H, H),
                    ("self_attn.o_proj", 24, H, H),
                    ("mlp.gate_proj", 25, I, H),
                    ("mlp.up_proj", 26, I, H),
                    ("mlp.down_proj", 27, H, I),
                ] {
                    tensors.extend(optiq_linear(
                        &format!("mtp.layers.0.{name}"),
                        seed,
                        rows,
                        cols,
                    ));
                }
                let tmp = write_sidecar(&tensors);
                let cpu_head = MtpHead::from_sidecar_with(
                    tmp.path(),
                    &model.config,
                    H,
                    &MtpLoadContext::cpu(),
                    MtpMoeQuant::Affine,
                )?;
                let model = model.with_mtp_sidecar(tmp.path())?;
                let head = model.mtp.as_ref().expect("invariant: tête chargée");
                assert!(matches!(head.layer.mlp, FeedForward::Dense(_)));
                assert!(matches!(head.fc.weight(), LinearWeight::Dense(_)));
                let linears = quantized_linears(head);
                assert_eq!(linears.len(), 7);
                for (index, (cpu_linear, linear)) in quantized_linears(&cpu_head)
                    .iter()
                    .zip(&linears)
                    .enumerate()
                {
                    let (cpu_w, w) = (affine(cpu_linear), affine(linear));
                    assert_eq!(w.packed_data(), cpu_w.packed_data(), "linéaire {index}");
                    assert_eq!(*w.scales_f32(), *cpu_w.scales_f32(), "linéaire {index}");
                    assert_eq!(views(w).is_some(), single_copy, "linéaire {index}");
                }
                assert_eq!(cpu_forward(&model, head)?, cpu_forward(&model, &cpu_head)?);
                Ok(())
            }
        }

        #[test]
        fn ambiguous_colliding_or_incomplete_sidecars_are_rejected() {
            let mut both = moe_tensors();
            both.push((
                "mtp.layers.0.mlp.gate_proj.weight".into(),
                tensor(&[I, H], |_| 0.0),
            ));
            assert!(matches!(
                load(&both, MtpMoeQuant::Dense),
                Err(InferError::Config(_))
            ));

            let mut collision = moe_tensors();
            collision.push((
                "language_model.mtp.fc.weight".into(),
                tensor(&[H, 2 * H], |_| 0.0),
            ));
            assert!(matches!(
                load(&collision, MtpMoeQuant::Dense),
                Err(InferError::Config(_))
            ));

            // Norme finale trop courte : refusée avant tout encodage GPU, dont
            // les kernels liraient `hidden` coefficients hors du buffer.
            let mut short = moe_tensors();
            for (name, raw) in &mut short {
                if name == "mtp.norm.weight" {
                    *raw = tensor(&[1], |_| 0.0);
                }
            }
            assert!(matches!(
                load(&short, MtpMoeQuant::Dense),
                Err(InferError::Dimension(_))
            ));

            let mut wide = moe_tensors();
            for (name, raw) in &mut wide {
                if name == "mtp.layers.0.self_attn.o_proj.weight" {
                    *raw = tensor(&[H, 2 * H], ramp);
                }
            }
            assert!(matches!(
                load(&wide, MtpMoeQuant::Dense),
                Err(InferError::Dimension(_))
            ));

            let mut missing = moe_tensors();
            missing.retain(|(name, _)| name != "mtp.layers.0.mlp.experts.down_proj");
            assert!(load(&missing, MtpMoeQuant::Dense).is_err());
        }

        /// Header tronqué, longueur de header hors fichier, offsets invalides
        /// et fichier raccourci : erreur, jamais de panique ni de pages à zéro
        /// acceptées comme poids.
        #[test]
        fn partial_or_corrupt_sidecar_reads_are_errors() {
            use super::super::LazySidecar;
            use std::io::Write;

            let config = moe_config(Some(16), E);
            let load_path = |path: &std::path::Path| {
                MtpHead::from_sidecar_with(
                    path,
                    &config,
                    H,
                    &MtpLoadContext::cpu(),
                    MtpMoeQuant::Affine,
                )
            };
            let write = |bytes: &[u8]| {
                let mut tmp =
                    tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
                tmp.write_all(bytes).expect("invariant: écriture");
                tmp
            };
            let valid = std::fs::read(write_sidecar(&f32_typed(&moe_tensors())).path())
                .expect("invariant: lecture sidecar");
            let header_len = usize::try_from(u64::from_le_bytes(
                valid[..8].try_into().expect("invariant: 8 octets"),
            ))
            .expect("invariant: header usize");

            // Moins de 8 octets : pas même la longueur du header.
            assert!(load_path(write(&valid[..4]).path()).is_err());
            // Header annoncé plus long que le fichier.
            let mut huge = valid.clone();
            huge[..8].copy_from_slice(&(valid.len() as u64).to_le_bytes());
            assert!(load_path(write(&huge).path()).is_err());
            // Header coupé en plein JSON.
            assert!(load_path(write(&valid[..8 + header_len / 2]).path()).is_err());
            // Données tronquées : les offsets ne couvrent plus le fichier.
            assert!(load_path(write(&valid[..valid.len() - 16]).path()).is_err());
            // Offsets invalides : un tenseur annonce une fin au-delà des données.
            let header =
                std::str::from_utf8(&valid[8..8 + header_len]).expect("invariant: header UTF-8");
            let data_len = valid.len() - 8 - header_len;
            let needle = format!(",{data_len}]");
            assert!(
                header.contains(&needle),
                "fin des données attendue dans le header"
            );
            let bad = header.replacen(&needle, &format!(",{}]", data_len + 64), 1);
            let mut corrupt = Vec::new();
            corrupt.extend_from_slice(&(bad.len() as u64).to_le_bytes());
            corrupt.extend_from_slice(bad.as_bytes());
            corrupt.extend_from_slice(&valid[8 + header_len..]);
            assert!(load_path(write(&corrupt).path()).is_err());

            // Fichier raccourci après l'ouverture, avant la lecture d'un expert.
            let tmp = write(&valid);
            let sidecar = LazySidecar::open(tmp.path()).expect("invariant: sidecar valide");
            let st = SafeTensors::deserialize(&sidecar.bytes).expect("invariant: header valide");
            let resolve: std::collections::HashMap<String, String> = st
                .names()
                .into_iter()
                .map(|name| {
                    (
                        super::super::normalize_mtp_sidecar_key(name),
                        name.to_string(),
                    )
                })
                .collect();
            let offset = sidecar
                .data_offset(&resolve, super::super::MTP_MOE_MARKER)
                .expect("invariant: offset des experts");
            std::fs::OpenOptions::new()
                .write(true)
                .open(tmp.path())
                .and_then(|file| file.set_len(offset as u64 + 8))
                .expect("invariant: troncature");
            assert!(
                sidecar.read(offset, 64).is_err(),
                "pages absentes lues comme zéro"
            );
        }
    }
}
