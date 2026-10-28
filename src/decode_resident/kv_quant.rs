//! Format du KV résident full-attn : dense (f32, bf16) ou quantifié affine.

use super::arena::alloc_tensor;
use super::gpu_position::to_u32;
use super::kernels::compile_kernels;
use super::utils::byte_offset;
use super::*;
use rayon::prelude::*;

/// Encodage d'UN tenseur résident K ou V.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvEncoding {
    F32,
    Bf16,
    /// Codes 8 bits, échelle et biais par groupe de valeurs.
    Q8,
    /// Codes 4 bits, deux par octet, échelle et biais par groupe de valeurs.
    Q4,
}

impl KvEncoding {
    /// Type d'élément du buffer de charge utile.
    pub(crate) const fn element(self) -> GpuElement {
        match self {
            Self::F32 => GpuElement::F32,
            Self::Bf16 => GpuElement::Bf16,
            Self::Q8 | Self::Q4 => GpuElement::U8,
        }
    }

    /// Indique un encodage quantifié (codes + échelles).
    pub(crate) const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::Q4)
    }

    /// Bits par code, `None` pour un encodage dense.
    pub(crate) const fn bits(self) -> Option<usize> {
        match self {
            Self::F32 | Self::Bf16 => None,
            Self::Q8 => Some(8),
            Self::Q4 => Some(4),
        }
    }

    /// Éléments du buffer de charge utile pour `cells` valeurs (octets pour un
    /// encodage quantifié), `None` si `cells` ne remplit pas des groupes
    /// entiers ou déborde.
    pub(crate) fn payload_len(self, cells: usize) -> Option<usize> {
        match self.bits() {
            None => Some(cells),
            Some(bits) if cells % KV_QUANT_GROUP == 0 => {
                cells.checked_mul(bits).map(|total| total / 8)
            }
            Some(_) => None,
        }
    }
}

/// Format du KV résident d'une couche full-attn.
///
/// K et V portent chacun leur encodage : une erreur sur K déforme les scores
/// avant le softmax, une erreur sur V seulement la moyenne pondérée, d'où le
/// mode mixte K8V4.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum KvFormat {
    F32,
    Bf16,
    Q8,
    K8V4,
    Q4,
}

impl KvFormat {
    /// Encodage des clés.
    pub(crate) const fn keys(self) -> KvEncoding {
        match self {
            Self::F32 => KvEncoding::F32,
            Self::Bf16 => KvEncoding::Bf16,
            Self::Q8 | Self::K8V4 => KvEncoding::Q8,
            Self::Q4 => KvEncoding::Q4,
        }
    }

    /// Encodage des valeurs.
    pub(crate) const fn values(self) -> KvEncoding {
        match self {
            Self::F32 => KvEncoding::F32,
            Self::Bf16 => KvEncoding::Bf16,
            Self::Q8 => KvEncoding::Q8,
            Self::K8V4 | Self::Q4 => KvEncoding::Q4,
        }
    }

    /// Indique un format quantifié (K et V quantifiés ensemble).
    pub(crate) const fn is_quantized(self) -> bool {
        matches!(self, Self::Q8 | Self::K8V4 | Self::Q4)
    }
}

/// Valeurs par groupe de quantification, contiguës le long de `head_dim`
/// (d256 : 4 groupes par tête et par ligne, d128 : 2). Un groupe ne chevauche
/// jamais deux têtes puisque `head_dim` est un multiple de 64.
pub(crate) const KV_QUANT_GROUP: usize = 64;

/// Code affine d'une valeur, miroir exact de `kv_code` du kernel : échelle
/// nulle ou NaN, valeur NaN ou sous le biais → 0 ; saturation à `levels`. Les
/// comparaisons explicites rendent ces cas définis à l'identique sur CPU et GPU.
/// NOTE: une valeur ±inf rend l'échelle infinie et toute la relecture du groupe
/// NaN (`fma(0, inf, b)`), là où bf16 ne perd qu'une valeur ; un K/V infini
/// signale déjà un modèle divergé.
fn quant_code(x: f32, scale: f32, bias: f32, levels: u8) -> u8 {
    let t = (x - bias) / scale;
    if scale > 0.0 && t > 0.0 {
        if t >= f32::from(levels) {
            levels
        } else {
            // `t` est dans ]0, levels[ : l'arrondi tient dans un u8.
            t.round_ties_even() as u8
        }
    } else {
        0
    }
}

/// Quantifie un groupe de [`KV_QUANT_GROUP`] valeurs : écrit les codes
/// empaquetés dans `out` (64 octets en Q8, 32 en Q4, `q[2i]` dans le quartet
/// bas) et renvoie `[échelle, biais]`.
///
/// min/max séquentiels de `j = 0` à 63 avec comparaisons strictes, comme le
/// kernel : un NaN en tête se propage à l'échelle (codes tous nuls), ailleurs
/// il est ignoré par les bornes et codé 0.
fn quantize_group(bits: usize, x: &[f32], out: &mut [u8]) -> [f32; 2] {
    let levels: u8 = if bits == 8 { 255 } else { 15 };
    let mut lo = x[0];
    let mut hi = x[0];
    for &value in &x[1..] {
        if value < lo {
            lo = value;
        }
        if value > hi {
            hi = value;
        }
    }
    let scale = (hi - lo) / f32::from(levels);
    if bits == 8 {
        for (code, &value) in out.iter_mut().zip(x) {
            *code = quant_code(value, scale, lo, levels);
        }
    } else {
        for (byte, pair) in out.iter_mut().zip(x.chunks_exact(2)) {
            *byte = quant_code(pair[0], scale, lo, levels)
                | (quant_code(pair[1], scale, lo, levels) << 4);
        }
    }
    [scale, lo]
}

/// Vérifie les longueurs d'un bloc de lignes quantifiées et renvoie les bits
/// par code.
fn check_quant_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    values: usize,
    payload: usize,
    scales: usize,
) -> Result<usize> {
    let bits = encoding
        .bits()
        .ok_or_else(|| InferError::Config(format!("KV quantifié: encodage {encoding:?} dense")))?;
    if kv_dim == 0 || kv_dim % KV_QUANT_GROUP != 0 || values % kv_dim != 0 {
        return Err(InferError::Dimension(format!(
            "KV quantifié: {values} valeurs pour des lignes de {kv_dim} (groupes de {KV_QUANT_GROUP})"
        )));
    }
    let groups = values / KV_QUANT_GROUP;
    if encoding.payload_len(values) != Some(payload) || groups.checked_mul(2) != Some(scales) {
        return Err(InferError::Dimension(format!(
            "KV quantifié: {values} valeurs, charge utile {payload} octets, {scales} échelles"
        )));
    }
    Ok(bits)
}

/// Oracle CPU : quantifie des lignes `[rows, kv_dim]` vers `payload` (codes)
/// et `scales` (`[rows, kv_dim / 64]` paires échelle, biais), même disposition
/// que le KV résident. Parallèle par groupe, chacun étant indépendant.
///
/// # Errors
///
/// Renvoie une erreur si l'encodage est dense ou si les longueurs ne forment
/// pas des lignes entières de groupes complets.
pub(super) fn quantize_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    values: &[f32],
    payload: &mut [u8],
    scales: &mut [f32],
) -> Result<()> {
    let bits = check_quant_rows(encoding, kv_dim, values.len(), payload.len(), scales.len())?;
    values
        .par_chunks(KV_QUANT_GROUP)
        .zip(payload.par_chunks_mut(KV_QUANT_GROUP * bits / 8))
        .zip(scales.par_chunks_mut(2))
        .for_each(|((x, out), scale_bias)| {
            scale_bias.copy_from_slice(&quantize_group(bits, x, out));
        });
    Ok(())
}

/// Oracle CPU : déquantifie des lignes produites par [`quantize_rows`] (ou
/// par le kernel) vers `out`, avec `fma(code, échelle, biais)` comme le kernel
/// d'attention : les valeurs relues sont exactement celles que lit le GPU.
///
/// # Errors
///
/// Mêmes conditions que [`quantize_rows`].
pub(super) fn dequantize_rows(
    encoding: KvEncoding,
    kv_dim: usize,
    payload: &[u8],
    scales: &[f32],
    out: &mut [f32],
) -> Result<()> {
    let bits = check_quant_rows(encoding, kv_dim, out.len(), payload.len(), scales.len())?;
    out.par_chunks_mut(KV_QUANT_GROUP)
        .zip(payload.par_chunks(KV_QUANT_GROUP * bits / 8))
        .zip(scales.par_chunks(2))
        .for_each(|((x, codes), scale_bias)| {
            let (scale, bias) = (scale_bias[0], scale_bias[1]);
            for (index, value) in x.iter_mut().enumerate() {
                let code = if bits == 8 {
                    codes[index]
                } else {
                    (codes[index / 2] >> ((index % 2) * 4)) & 0x0f
                };
                *value = f32::from(code).mul_add(scale, bias);
            }
        });
    Ok(())
}

/// Kernels de quantification du KV, compilés SANS fast-math : division et
/// arrondi reproduisent [`quantize_rows`] bit à bit (codes et échelles).
///
/// Un thread par (ligne, groupe de 64) écrit les codes ET l'échelle de son
/// groupe : aucune écriture partagée entre threads (quartets Q4 compris), et
/// une ligne réécrite après troncature (rejet MTP) ne garde rien de l'ancienne.
const KV_QUANTIZE_KERNELS: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Miroir de `quant_code` (kv_quant.rs).
inline uint kv_code(float x, float s, float b, float levels) {
    const float t = (x - b) / s;
    if (s > 0.0f && t > 0.0f) {
        return (t >= levels) ? uint(levels) : uint(rint(t));
    }
    return 0u;
}

// p = [cols, in_stride, rows, 0] ; `payload` et `scales` sont liés à la
// première ligne cible, `input` à la première ligne source (stride en floats).
template <uint BITS>
inline void kv_quantize_rows(
    device const float* input,
    device uchar* payload,
    device float2* scales,
    constant uint4& p,
    uint gid
) {
    constexpr uint G = 64u;
    constexpr uint PER_BYTE = 8u / BITS;
    constexpr float LEVELS = float((1u << BITS) - 1u);
    const uint groups = p.x / G;
    if (gid >= groups * p.z) { return; }
    const uint row = gid / groups;
    const uint g = gid - row * groups;
    // Index 64 bits : `row * stride` déborde un uint au-delà de ~1M lignes de 4096.
    const device float* x = input + ulong(row) * p.y + g * G;
    float lo = x[0];
    float hi = x[0];
    for (uint j = 1u; j < G; ++j) {
        const float v = x[j];
        if (v < lo) { lo = v; }
        if (v > hi) { hi = v; }
    }
    const float s = (hi - lo) / LEVELS;
    scales[ulong(row) * groups + g] = float2(s, lo);
    device uchar* out = payload + (ulong(row) * p.x + g * G) / PER_BYTE;
    for (uint j = 0u; j < G; j += PER_BYTE) {
        uint byte = 0u;
        for (uint c = 0u; c < PER_BYTE; ++c) {
            byte |= kv_code(x[j + c], s, lo, LEVELS) << (c * BITS);
        }
        out[j / PER_BYTE] = uchar(byte);
    }
}

kernel void kv_quantize_rows_q8(
    device const float* input [[buffer(0)]],
    device uchar* payload [[buffer(1)]],
    device float2* scales [[buffer(2)]],
    constant uint4& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    kv_quantize_rows<8u>(input, payload, scales, p, gid);
}

kernel void kv_quantize_rows_q4(
    device const float* input [[buffer(0)]],
    device uchar* payload [[buffer(1)]],
    device float2* scales [[buffer(2)]],
    constant uint4& p [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    kv_quantize_rows<4u>(input, payload, scales, p, gid);
}
"#;

/// Passe 1 de l'attention 2-passes split-K sur KV quantifié, dérivée de
/// `attention_decode_2pass_1_rows_dDHEAD_SUFFIX` (rows.rs) : mêmes grille,
/// partiels et passe 2. Chaque lane lit ses `QK` codes bruts et applique
/// l'échelle et le biais de son groupe hors des produits (cf. le kernel) :
/// l'arrondi diffère donc de `dequantize_rows` à l'ulp près. Le mono-ligne
/// est le cas `rows = 1` (`tidtg.z = 0`). Une lane lit un groupe entier
/// d'échelle (64 ≥ QK) : une seule paire par lane et par ligne.
const KV_QUANT_ATTENTION_TEMPLATE: &str = r#"
kernel void attention_decode_2pass_1_q_dDHEAD_FMT(
    device const float* q [[buffer(0)]],
    device const uchar* keys [[buffer(1)]],
    device const uchar* values [[buffer(2)]],
    device float* partials [[buffer(3)]],
    device float* sums [[buffer(4)]],
    device float* maxs [[buffer(5)]],
    constant uint4& dims [[buffer(6)]],
    device const float2* key_scales [[buffer(7)]],
    device const float2* value_scales [[buffer(8)]],
    uint3 tidtg [[thread_position_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_lid [[thread_index_in_simdgroup]]
) {
    constexpr uint BD = 32u;
    constexpr uint D = DHEADu;
    constexpr uint QK = D / BD;
    constexpr uint G = 64u;
    const uint q_heads = dims.x;
    const uint kv_heads = dims.y;
    const uint blocks = dims.w;
    const uint gqa = q_heads / kv_heads;
    const uint kv_head = tid.x;
    const uint block_idx = tid.y;
    const uint q_in_group = tidtg.y;
    const uint row = tidtg.z;
    const uint len = dims.z + row;
    const uint q_head = kv_head * gqa + q_in_group;
    if (q_head >= q_heads || block_idx >= blocks) { return; }
    const uint slot = row * q_heads + q_head;
    const uint kv_dim = kv_heads * D;
    const uint groups = kv_dim / G;
    const float scale = rsqrt(float(D));

    float ql[QK];
    {
        const device float* q_ = q + slot * D + simd_lid * QK;
        for (uint j = 0u; j < QK; ++j) { ql[j] = scale * q_[j]; }
    }
    const uint lane = kv_head * D + simd_lid * QK;
    const device uchar* k_ = keys + (block_idx * kv_dim + lane) * K_BITSu / 8u;
    const device uchar* v_ = values + (block_idx * kv_dim + lane) * V_BITSu / 8u;
    const device float2* ks_ = key_scales + block_idx * groups + lane / G;
    const device float2* vs_ = value_scales + block_idx * groups + lane / G;
    const uint k_step = blocks * kv_dim * K_BITSu / 8u;
    const uint v_step = blocks * kv_dim * V_BITSu / 8u;
    const uint s_step = blocks * groups;

    // Échelle et biais sortent des produits : `q·(s·c + b) = s·(q·c) + b·Σq`
    // pour K, et `p·(s·c + b)` s'accumule en `(p·s)·c` plus un terme de biais
    // commun aux QK valeurs de la lane (un seul groupe de 64 par lane). La
    // boucle est bornée par les instructions de déquantification, pas par les
    // octets lus : ce repliement en retire une sur deux.
    float q_sum = 0.0f;
    for (uint j = 0u; j < QK; ++j) { q_sum += ql[j]; }
    float o[QK];
    for (uint j = 0u; j < QK; ++j) { o[j] = 0.0f; }
    float o_bias = 0.0f;
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    for (uint i = block_idx; i < len; i += blocks) {
        float k[QK];
        kv_codes_K_BITS(k_, k);
        const float2 ks = *ks_;
        float dot = 0.0f;
        for (uint j = 0u; j < QK; ++j) { dot += ql[j] * k[j]; }
        const float score = simd_sum(fma(ks.x, dot, ks.y * q_sum));
        const float new_max = max(max_score, score);
        const float factor = fast::exp(max_score - new_max);
        const float exp_score = fast::exp(score - new_max);
        max_score = new_max;
        sum_exp = sum_exp * factor + exp_score;
        float v[QK];
        kv_codes_V_BITS(v_, v);
        const float2 vs = *vs_;
        const float weight = exp_score * vs.x;
        for (uint j = 0u; j < QK; ++j) { o[j] = fma(weight, v[j], o[j] * factor); }
        o_bias = fma(exp_score, vs.y, o_bias * factor);
        k_ += k_step;
        v_ += v_step;
        ks_ += s_step;
        vs_ += s_step;
    }
    for (uint j = 0u; j < QK; ++j) { o[j] += o_bias; }

    if (simd_lid == 0u) {
        sums[slot * blocks + block_idx] = sum_exp;
        maxs[slot * blocks + block_idx] = max_score;
    }
    device float* out_ = partials + (slot * blocks + block_idx) * D + simd_lid * QK;
    for (uint j = 0u; j < QK; ++j) { out_[j] = o[j]; }
}
"#;

/// Formats quantifiés et suffixe de leur kernel d'attention.
const KV_QUANT_FORMATS: [(KvFormat, &str); 3] = [
    (KvFormat::Q8, "q8"),
    (KvFormat::K8V4, "k8v4"),
    (KvFormat::Q4, "q4"),
];

/// Codes bruts (en flottants) des `qk` valeurs de `bits` bits d'une lane, lus
/// en un chargement aligné ; code `c` d'un mot au décalage `c·bits`
/// (petit-boutiste : quartet bas = indice pair).
fn lane_codes_source(bits: usize, qk: usize) -> String {
    // Un seul chargement vectoriel par lane : le coût de la boucle suit le
    // nombre de lectures, pas leur taille (mesuré : Q4 aussi lent que Q8).
    let (load, words) = match bits * qk {
        16 => ("ushort", 1),
        32 => ("uint", 1),
        64 => ("uint2", 2),
        _ => ("uint4", 4),
    };
    let per_word = qk / words;
    let mask = (1_u32 << bits) - 1;
    let word = |i: usize| {
        if words == 1 {
            String::new()
        } else {
            format!("[{i}]")
        }
    };
    let mut body = String::new();
    for i in 0..words {
        for c in 0..per_word {
            body.push_str(&format!(
                "    x[{}] = float((uint(w{}) >> {}u) & {mask}u);\n",
                i * per_word + c,
                word(i),
                c * bits
            ));
        }
    }
    format!(
        "
inline void kv_codes_{bits}(const device uchar* p, thread float* x) {{
    const {load} w = *(const device {load}*)p;
{body}}}
"
    )
}

/// Source des trois kernels d'attention quantifiée pour `head_dim`.
fn attention_source(head_dim: usize) -> String {
    let qk = head_dim / 32;
    let mut source = String::from(
        "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n",
    );
    source.push_str(&lane_codes_source(8, qk));
    source.push_str(&lane_codes_source(4, qk));
    for (format, suffix) in KV_QUANT_FORMATS {
        let bits = |encoding: KvEncoding| encoding.bits().unwrap_or(8).to_string();
        source.push_str(
            &KV_QUANT_ATTENTION_TEMPLATE
                .replace("DHEAD", &head_dim.to_string())
                .replace("FMT", suffix)
                .replace("K_BITS", &bits(format.keys()))
                .replace("V_BITS", &bits(format.values())),
        );
    }
    source
}

/// Pipelines du KV quantifié, compilés au premier KV quantifié construit.
#[derive(Debug)]
pub(super) struct KvQuantKernels {
    quantize_q8: ComputePipelineState,
    quantize_q4: ComputePipelineState,
    /// Passe 1 de l'attention par `(head_dim, format)`.
    attention: Vec<((usize, KvFormat), ComputePipelineState)>,
}

impl DecodeResidentState {
    pub(super) fn kv_quant_kernels(&self) -> Result<&KvQuantKernels> {
        if let Some(kernels) = self.kv_quant_kernels.get() {
            return Ok(kernels);
        }
        let mut quantize = compile_kernels(
            &self.device,
            KV_QUANTIZE_KERNELS,
            &["kv_quantize_rows_q8", "kv_quantize_rows_q4"],
            false,
        )?
        .into_iter();
        let (Some(quantize_q8), Some(quantize_q4)) = (quantize.next(), quantize.next()) else {
            return Err(InferError::Metal(
                "kernels de quantification KV incomplets".to_string(),
            ));
        };
        let mut attention = Vec::new();
        for head_dim in [128, 256] {
            let names: Vec<String> = KV_QUANT_FORMATS
                .iter()
                .map(|(_, suffix)| format!("attention_decode_2pass_1_q_d{head_dim}_{suffix}"))
                .collect();
            let names: Vec<&str> = names.iter().map(String::as_str).collect();
            // Fast-math comme les kernels d'attention denses (`fast::exp` explicite).
            let pipelines =
                compile_kernels(&self.device, &attention_source(head_dim), &names, true)?;
            for ((format, _), pipeline) in KV_QUANT_FORMATS.iter().zip(pipelines) {
                attention.push(((head_dim, *format), pipeline));
            }
        }
        let kernels = KvQuantKernels {
            quantize_q8,
            quantize_q4,
            attention,
        };
        Ok(self.kv_quant_kernels.get_or_init(|| kernels))
    }

    /// Alloue les métadonnées d'un KV `format` de `cells` valeurs par tenseur.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si le format est dense, si `head_dim` n'a pas de
    /// kernel d'attention quantifiée, ou si la compilation échoue.
    pub(super) fn kv_quant(
        &self,
        format: KvFormat,
        head_dim: usize,
        cells: usize,
    ) -> Result<KvQuant> {
        let attention = self
            .kv_quant_kernels()?
            .attention
            .iter()
            .find(|(key, _)| *key == (head_dim, format))
            .map(|(_, pipeline)| pipeline.clone())
            .ok_or_else(|| {
                InferError::Config(format!(
                    "KV {format:?}: head_dim={head_dim} non supporté (128 ou 256)"
                ))
            })?;
        Ok(KvQuant {
            keys: self.quant_tensor(format.keys(), cells)?,
            values: self.quant_tensor(format.values(), cells)?,
            attention,
        })
    }

    /// Alloue les échelles d'un tenseur quantifié de `cells` valeurs.
    fn quant_tensor(&self, encoding: KvEncoding, cells: usize) -> Result<QuantTensor> {
        let kernels = self.kv_quant_kernels()?;
        let quantize = match encoding {
            KvEncoding::Q8 => kernels.quantize_q8.clone(),
            KvEncoding::Q4 => kernels.quantize_q4.clone(),
            KvEncoding::F32 | KvEncoding::Bf16 => {
                return Err(InferError::Config(format!(
                    "KV quantifié: encodage {encoding:?} dense"
                )));
            }
        };
        let scale_cells = (cells / KV_QUANT_GROUP)
            .checked_mul(2)
            .ok_or_else(|| InferError::Dimension("échelles KV débordent".to_string()))?;
        Ok(QuantTensor {
            encoding,
            scales: alloc_tensor(&self.device, self.options, scale_cells, GpuElement::F32)?,
            quantize,
        })
    }
}

/// Métadonnées des tenseurs K et V d'un KV quantifié.
#[derive(Debug)]
pub(super) struct KvQuant {
    pub(super) keys: QuantTensor,
    pub(super) values: QuantTensor,
    /// Passe 1 de l'attention 2-passes pour ce format et ce `head_dim`.
    pub(super) attention: ComputePipelineState,
}

/// Métadonnées d'UN tenseur K ou V quantifié : échelles `[capacity, kv_dim/64]`
/// en paires `float2` (échelle, biais) et kernel d'écriture. La charge utile
/// (codes) reste le `GpuTensor` u8 de l'état.
#[derive(Debug)]
pub(super) struct QuantTensor {
    encoding: KvEncoding,
    scales: GpuTensor,
    quantize: ComputePipelineState,
}

impl QuantTensor {
    pub(super) fn scales(&self) -> &GpuTensor {
        &self.scales
    }

    /// Offsets (octets de charge utile, octets d'échelles) de la ligne `row`.
    fn row_offsets(&self, row: usize, kv_dim: usize) -> Result<(u64, u64)> {
        let cells = row
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("ligne KV quantifiée déborde".to_string()))?;
        let payload = self
            .encoding
            .payload_len(cells)
            .ok_or_else(|| InferError::Dimension("offset KV quantifié invalide".to_string()))?;
        Ok((
            byte_offset(payload, GpuElement::U8, "offset KV quantifié")?,
            byte_offset(
                cells / KV_QUANT_GROUP * 2,
                GpuElement::F32,
                "offset échelles KV",
            )?,
        ))
    }

    /// Encode la quantification de `rows` lignes : la ligne `r` lit
    /// `input[input_offset + r·in_stride ..]` (stride en floats) et écrit la
    /// ligne `first_row + r` de `payload` et des échelles. Bornes vérifiées par
    /// l'appelant (capacité).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si une dimension déborde.
    #[expect(
        clippy::too_many_arguments,
        reason = "encodage Metal: source + offset + stride, cible + lignes"
    )]
    pub(super) fn encode_rows(
        &self,
        encoder: &ComputeCommandEncoderRef,
        payload: &GpuTensor,
        input: &BufferRef,
        input_offset: u64,
        in_stride: usize,
        first_row: usize,
        rows: usize,
        kv_dim: usize,
    ) -> Result<()> {
        let (payload_offset, scales_offset) = self.row_offsets(first_row, kv_dim)?;
        let threads = rows
            .checked_mul(kv_dim / KV_QUANT_GROUP)
            .ok_or_else(|| InferError::Dimension("quantification KV déborde".to_string()))?;
        let p = [
            to_u32(kv_dim, "quantification KV kv_dim")?,
            to_u32(in_stride, "quantification KV stride")?,
            to_u32(rows, "quantification KV lignes")?,
            0,
        ];
        to_u32(threads, "quantification KV threads")?;
        encoder.set_compute_pipeline_state(&self.quantize);
        encoder.set_buffer(0, Some(input), input_offset);
        encoder.set_buffer(1, Some(payload.buffer()), payload_offset);
        encoder.set_buffer(2, Some(self.scales.buffer()), scales_offset);
        encoder.set_bytes(
            3,
            std::mem::size_of::<[u32; 4]>() as u64,
            p.as_ptr().cast::<c_void>(),
        );
        let width = self.quantize.thread_execution_width().max(1);
        crate::metal_backend::profile_dispatch();
        encoder.dispatch_threads(
            MTLSize::new(threads as u64, 1, 1),
            MTLSize::new(width, 1, 1),
        );
        crate::metal_backend::post_dispatch_barrier(encoder);
        Ok(())
    }

    /// Quantifie sur le CPU (oracle [`quantize_rows`]) les lignes `values` vers
    /// les lignes `first_row..` de `payload` et des échelles, avant tout commit
    /// qui les lirait (même discipline que `write_f32_at`).
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les lignes sortent du buffer ou si les longueurs
    /// ne forment pas des lignes entières.
    #[expect(
        unsafe_code,
        reason = "écriture CPU d'un MTLBuffer StorageModeShared avant commit"
    )]
    pub(super) fn write_rows(
        &self,
        payload: &GpuTensor,
        first_row: usize,
        kv_dim: usize,
        values: &[f32],
    ) -> Result<()> {
        if values.is_empty() {
            return Ok(());
        }
        let (payload_range, scales_range) =
            self.ranges(payload, first_row, kv_dim, values.len())?;
        let payload_ptr = payload.buffer().contents().cast::<u8>();
        let scales_ptr = self.scales.buffer().contents().cast::<f32>();
        if payload_ptr.is_null() || scales_ptr.is_null() {
            return Err(InferError::Metal(
                "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
            ));
        }
        // SAFETY: buffers StorageModeShared distincts, plages vérifiées par
        // `ranges` dans leurs longueurs logiques ; écriture avant le commit du
        // command buffer qui lira ces lignes, aucun autre accès CPU concurrent.
        let (codes, scales) = unsafe {
            (
                std::slice::from_raw_parts_mut(
                    payload_ptr.add(payload_range.start),
                    payload_range.len(),
                ),
                std::slice::from_raw_parts_mut(
                    scales_ptr.add(scales_range.start),
                    scales_range.len(),
                ),
            )
        };
        quantize_rows(self.encoding, kv_dim, values, codes, scales)
    }

    /// Relit et déquantifie `rows` lignes à partir de `first_row`, après la fin
    /// du command buffer qui les a écrites.
    ///
    /// # Errors
    ///
    /// Renvoie une erreur si les lignes sortent du buffer.
    #[expect(
        unsafe_code,
        reason = "lecture CPU d'un MTLBuffer StorageModeShared après wait_until_completed"
    )]
    pub(super) fn read_rows(
        &self,
        payload: &GpuTensor,
        first_row: usize,
        rows: usize,
        kv_dim: usize,
    ) -> Result<Vec<f32>> {
        let cells = rows
            .checked_mul(kv_dim)
            .ok_or_else(|| InferError::Dimension("lecture KV quantifié déborde".to_string()))?;
        if cells == 0 {
            return Ok(Vec::new());
        }
        let (payload_range, scales_range) = self.ranges(payload, first_row, kv_dim, cells)?;
        let payload_ptr = payload.buffer().contents().cast::<u8>();
        let scales_ptr = self.scales.buffer().contents().cast::<f32>();
        if payload_ptr.is_null() || scales_ptr.is_null() {
            return Err(InferError::Metal(
                "MTLBuffer StorageModeShared sans pointeur CPU".to_string(),
            ));
        }
        // SAFETY: plages vérifiées par `ranges`, command buffer écrivain terminé ;
        // lecture seule sans écriture concurrente.
        let (codes, scales) = unsafe {
            (
                std::slice::from_raw_parts(
                    payload_ptr.add(payload_range.start),
                    payload_range.len(),
                ),
                std::slice::from_raw_parts(scales_ptr.add(scales_range.start), scales_range.len()),
            )
        };
        let mut out = vec![0.0_f32; cells];
        dequantize_rows(self.encoding, kv_dim, codes, scales, &mut out)?;
        Ok(out)
    }

    /// Plages (octets de codes, floats d'échelles) de `cells` valeurs à partir
    /// de la ligne `first_row`, vérifiées dans les longueurs logiques.
    fn ranges(
        &self,
        payload: &GpuTensor,
        first_row: usize,
        kv_dim: usize,
        cells: usize,
    ) -> Result<(std::ops::Range<usize>, std::ops::Range<usize>)> {
        let invalid = || {
            InferError::Dimension(format!(
                "KV quantifié: {cells} valeurs depuis la ligne {first_row} hors buffer"
            ))
        };
        if kv_dim == 0 || kv_dim % KV_QUANT_GROUP != 0 || cells % kv_dim != 0 {
            return Err(invalid());
        }
        let start = first_row.checked_mul(kv_dim).ok_or_else(invalid)?;
        let end = start.checked_add(cells).ok_or_else(invalid)?;
        let code = |cells| self.encoding.payload_len(cells).ok_or_else(invalid);
        let payload_range = code(start)?..code(end)?;
        let scales_range = start / KV_QUANT_GROUP * 2..end / KV_QUANT_GROUP * 2;
        if payload_range.end > payload.len() || scales_range.end > self.scales.len() {
            return Err(invalid());
        }
        Ok((payload_range, scales_range))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize(encoding: KvEncoding, kv_dim: usize, values: &[f32]) -> (Vec<u8>, Vec<f32>) {
        let bytes = encoding.payload_len(values.len()).unwrap_or(0);
        let mut payload = vec![0_u8; bytes];
        let mut scales = vec![0.0_f32; values.len() / KV_QUANT_GROUP * 2];
        quantize_rows(encoding, kv_dim, values, &mut payload, &mut scales)
            .expect("invariant: longueurs de test cohérentes");
        (payload, scales)
    }

    fn roundtrip(encoding: KvEncoding, kv_dim: usize, values: &[f32]) -> Vec<f32> {
        let (payload, scales) = quantize(encoding, kv_dim, values);
        let mut out = vec![0.0_f32; values.len()];
        dequantize_rows(encoding, kv_dim, &payload, &scales, &mut out)
            .expect("invariant: longueurs de test cohérentes");
        out
    }

    #[test]
    fn aller_retour_borne_par_demi_pas() {
        let kv_dim = 256;
        let values: Vec<f32> = (0..4 * kv_dim)
            .map(|i| (i as f32 * 0.731).sin() * (1.0 + (i / 64) as f32))
            .collect();
        for (encoding, levels) in [(KvEncoding::Q8, 255.0_f32), (KvEncoding::Q4, 15.0)] {
            let out = roundtrip(encoding, kv_dim, &values);
            for (group, (x, y)) in values
                .chunks(KV_QUANT_GROUP)
                .zip(out.chunks(KV_QUANT_GROUP))
                .enumerate()
            {
                let lo = x.iter().copied().fold(f32::INFINITY, f32::min);
                let hi = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let half_step = (hi - lo) / levels / 2.0;
                for (a, b) in x.iter().zip(y) {
                    assert!(
                        (a - b).abs() <= half_step * 1.001 + 1e-6,
                        "{encoding:?} groupe {group}: {a} -> {b} (demi-pas {half_step})"
                    );
                }
                // Le minimum du groupe est codé 0 : restitué exactement.
                assert_eq!(y.iter().copied().fold(f32::INFINITY, f32::min), lo);
            }
        }
    }

    #[test]
    fn groupe_constant_et_disposition_q4() {
        let mut values = vec![3.5_f32; 128];
        // Second groupe : 0..15 répétés → codes exacts, quartet bas = indice pair.
        for (i, value) in values[64..].iter_mut().enumerate() {
            *value = (i % 16) as f32;
        }
        let (payload, scales) = quantize(KvEncoding::Q4, 64, &values);
        assert_eq!(&scales, &[0.0, 3.5, 1.0, 0.0], "échelle nulle, biais exact");
        assert!(payload[..32].iter().all(|&b| b == 0), "groupe constant → 0");
        assert_eq!(payload[32], 0x10, "q[0]=0 bas, q[1]=1 haut");
        assert_eq!(payload[39], 0xfe, "q[14]=14 bas, q[15]=15 haut");
        assert_eq!(
            roundtrip(KvEncoding::Q4, 64, &values),
            values,
            "valeurs sur la grille : exactes"
        );
    }

    #[test]
    fn valeurs_non_finies_definies() {
        let mut values: Vec<f32> = (0..64).map(|i| i as f32).collect();
        values[5] = f32::NAN;
        // NaN hors tête : ignoré par les bornes, codé 0.
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert_eq!(scales, [63.0 / 255.0, 0.0]);
        assert_eq!(payload[5], 0);
        assert_eq!(payload[63], 255);
        // NaN en tête : échelle NaN, tous les codes à 0.
        values[0] = f32::NAN;
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert!(scales[0].is_nan());
        assert!(payload.iter().all(|&b| b == 0));
        // Infini : échelle infinie, codes définis (0), aucun code hors plage.
        values[0] = f32::INFINITY;
        values[5] = 1.0;
        let (payload, scales) = quantize(KvEncoding::Q8, 64, &values);
        assert_eq!(scales, [f32::INFINITY, 1.0]);
        assert!(payload.iter().all(|&b| b == 0));
    }

    #[test]
    fn longueurs_incoherentes_refusees() {
        let mut payload = vec![0_u8; 64];
        let mut scales = vec![0.0_f32; 2];
        let values = vec![0.0_f32; 64];
        assert!(quantize_rows(KvEncoding::Bf16, 64, &values, &mut payload, &mut scales).is_err());
        assert!(quantize_rows(KvEncoding::Q8, 96, &values, &mut payload, &mut scales).is_err());
        assert!(quantize_rows(KvEncoding::Q4, 64, &values, &mut payload, &mut scales).is_err());
        assert_eq!(KvEncoding::Q4.payload_len(64), Some(32));
        assert_eq!(KvEncoding::Q8.payload_len(32), None);
        assert_eq!(KvEncoding::Bf16.payload_len(3), Some(3));
    }
}
