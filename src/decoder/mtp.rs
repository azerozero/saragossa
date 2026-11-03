//! Chargement et forward de la tête MTP.
//!
//! Supporte le format **OptiQ int4 + MLP dense** (27B-OptiQ : `.weight` u32 packé
//! 4-bit gs64 + `.scales`/`.biases` bf16, MLP `gate/up/down_proj`) en plus des
//! normes/`fc` bf16 (27B), et la tête **officielle transformers à MLP MoE**
//! (35B-A3B : experts fusionnés `gate_up_proj`, bf16), quantifiée à la charge
//! en affine gs64 comme le tronc, normes en `1 + weight`.

use super::*;
use crate::quantization::bytes_to_u32;
use crate::safetensor::bytes_to_dense_f32;
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
const MTP_QUANT_GROUP: usize = 64;

impl MtpHead {
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
        let bytes = std::fs::read(path.as_ref())
            .map_err(|source| InferError::Config(format!("lecture sidecar MTP: {source}")))?;
        let st = SafeTensors::deserialize(&bytes)
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
        let dims = MtpDims::new(config, hidden)?;
        match (
            resolve.contains_key(MTP_DENSE_MARKER),
            resolve.contains_key(MTP_MOE_MARKER),
        ) {
            (true, false) => Self::dense_from_tensors(&st, &resolve, config, &dims, context),
            (false, true) => {
                Self::moe_from_tensors(&st, &resolve, config, &dims, context, moe_quant)
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
            load_mtp_fused_experts(st, resolve, dims.hidden, quant)?;
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
    fn new(config: &CausalDecoderConfig, hidden: usize) -> Result<Self> {
        let head_dim = config
            .head_dim
            .ok_or_else(|| InferError::Config("tête MTP : head_dim absent de la config".into()))?;
        let product = |a: usize, b: usize| {
            a.checked_mul(b)
                .ok_or_else(|| InferError::Dimension("tête MTP : dimension déborde".into()))
        };
        let o_in = product(config.num_attention_heads, head_dim)?;
        // Q porte la gate de sortie concaténée par tête quand elle existe.
        let q_out = product(o_in, if config.attn_output_gate { 2 } else { 1 })?;
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
    resolve: &HashMap<String, String>,
    expected_hidden: usize,
    quant: MtpMoeQuant,
) -> Result<(Vec<LinearWeight>, Vec<LinearWeight>, Vec<LinearWeight>)> {
    let gate_up = st_view(st, resolve, MTP_MOE_MARKER)?;
    let down_key = "mtp.layers.0.mlp.experts.down_proj";
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
    // Experts indépendants : quantifiés en parallèle (≈ 800 M poids).
    let triples = (0..experts)
        .into_par_iter()
        .map(|expert| -> Result<_> {
            let bytes = &gate_up.data()[expert * gate_up_stride..(expert + 1) * gate_up_stride];
            let mut rows = bytes_to_dense_f32(bytes, gate_up.dtype(), MTP_MOE_MARKER)?;
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
            let bytes = &down.data()[expert * down_stride..(expert + 1) * down_stride];
            let rows = bytes_to_dense_f32(bytes, down.dtype(), down_key)?;
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
        use crate::{FeedForward, InferError, LinearWeight};
        use safetensors::{tensor::TensorView, Dtype};

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
            let views = tensors
                .iter()
                .map(|(name, (shape, bytes))| {
                    TensorView::new(Dtype::F32, shape.clone(), bytes)
                        .map(|view| (name.clone(), view))
                })
                .collect::<Result<Vec<_>, _>>()
                .expect("invariant: vues safetensors valides");
            let buffer = safetensors::serialize(views, None).expect("invariant: sérialisation");
            let tmp = tempfile::NamedTempFile::new().expect("invariant: fichier temporaire");
            std::fs::write(tmp.path(), buffer).expect("invariant: écriture sidecar");
            let config = CausalDecoderConfig {
                attn_output_gate: true,
                head_dim: Some(16),
                num_attention_heads: 4,
                num_key_value_heads: 4,
                num_experts: Some(E),
                num_experts_per_tok: 2,
                ..CausalDecoderConfig::default()
            };
            MtpHead::from_sidecar_with(tmp.path(), &config, H, &MtpLoadContext::cpu(), quant)
        }

        fn first(weight: &LinearWeight) -> Option<f32> {
            match weight {
                LinearWeight::Dense(tensor) => tensor.data().first().copied(),
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
    }
}
