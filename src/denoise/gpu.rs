//! Réseau DeepFilterNet3 en lot sur GPU Metal, pour le débruitage hors ligne.
//!
//! Les entrées de tout le fichier sont connues d'avance : convolutions et
//! linéaires traitent toutes les trames d'un morceau en un dispatch, et chaque
//! GRU parcourt sa boucle temporelle dans un seul noyau persistant (un
//! threadgroup, état caché en mémoire threadgroup). En flux, un hop coûte
//! ~35 dispatchs dépendants et le GPU ne gagne rien (`gpu_probe`).
//!
//! Identité au bit avec [`Math::Common`](super::layers::Math) sur CPU : chaque
//! sortie est calculée par un seul thread, dans l'ordre de sommation du CPU
//! (huit accumulateurs de `layers::dot`, boucles de `Conv::forward`), sans
//! fast-math ni contraction FMA, et `exact_math` remplace la libm des deux côtés.

use metal::{
    Buffer, CommandQueue, ComputeCommandEncoderRef, ComputePipelineState, Device,
    MTLResourceOptions, MTLSize,
};

use super::dsp::{NB_DF, NB_ERB};
use super::layers::{Act, Conv, GroupedLinear, Gru};
use super::network::{Network, SqueezedGru, CH, DF_OUT_CH, EMB, HIDDEN};
use crate::metal_backend::{commit_and_wait, read_f32_buffer_at, write_f32_buffer};
use crate::{InferError, Result};

/// Pas de réseau par command buffer : borne la mémoire (~100 Mo pour `c0`).
/// Les tests passent des frontières de morceau sur un clip court.
const CHUNK: usize = if cfg!(test) { 64 } else { 4096 };
/// Pas de contexte recalculés en tête de morceau : 4 trames de `c0` pour
/// `df_convp` (kt = 5), chacune sur 2 trames de spectre de plus (kt = 3).
const PREFIX: usize = 6;
const GATES: usize = 3 * HIDDEN;

const KERNELS: &str = r"
template <typename A, typename B>
static inline float dot8(A a, B b, int sb, int n) {
    #pragma METAL fp contract(off)
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        for (int l = 0; l < 8; ++l) { acc[l] += a[i + l] * b[(i + l) * sb]; }
    }
    float tail = 0.0f;
    for (; i < n; ++i) { tail += a[i] * b[i * sb]; }
    float s = acc[0];
    for (int l = 1; l < 8; ++l) { s += acc[l]; }
    return s + tail;
}

static inline float bn_act(float v, float s, float b, int act) {
    #pragma METAL fp contract(off)
    const float a = v * s + b;
    if (act == 2) { return dfx_sigmoid(a); }
    return a > 0.0f ? a : 0.0f;
}

// p = [cin, cout, groups, kt, kf, fstride, fin, fout, rows, zero_below, act]
kernel void dfn_conv(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* scale [[buffer(2)]], device const float* bias [[buffer(3)]],
    device float* y [[buffer(4)]], constant int* p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int cin = p[0], cout = p[1], groups = p[2], kt = p[3], kf = p[4];
    const int fs = p[5], fin = p[6], fout = p[7], rows = p[8], zb = p[9], act = p[10];
    const int t = int(gid.y), idx = int(gid.x);
    if (t >= rows || idx >= cout * fout) { return; }
    const int co = idx / fout, fo = idx % fout;
    const int cig = cin / groups, cog = cout / groups, g = co / cog, pad = kf / 2;
    float acc = 0.0f;
    for (int cl = 0; cl < cig; ++cl) {
        const int ci = g * cig + cl;
        for (int dt = 0; dt < kt; ++dt) {
            // Trame nulle : le CPU y ajoute des produits nuls, sans effet.
            const int tau = t - kt + 1 + dt;
            if (tau < zb) { continue; }
            device const float* xr = x + (tau * cin + ci) * fin;
            device const float* wk = w + ((co * cig + cl) * kt + dt) * kf;
            for (int k = 0; k < kf; ++k) {
                const int fi = fo * fs - pad + k;
                if (fi >= 0 && fi < fin) { acc += xr[fi] * wk[k]; }
            }
        }
    }
    if (act != 0) { acc = bn_act(acc, scale[co], bias[co], act); }
    y[(t * cout + co) * fout + fo] = acc;
}

kernel void dfn_convt(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* scale [[buffer(2)]], device const float* bias [[buffer(3)]],
    device float* y [[buffer(4)]], constant int* p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int cin = p[0], cout = p[1], groups = p[2], kf = p[4];
    const int fs = p[5], fin = p[6], fout = p[7], rows = p[8], act = p[10];
    const int t = int(gid.y), idx = int(gid.x);
    if (t >= rows || idx >= cout * fout) { return; }
    const int co = idx / fout, o = idx % fout;
    const int cig = cin / groups, cog = cout / groups, g = co / cog, col = co - g * cog;
    const int pad = kf / 2;
    float acc = 0.0f;
    for (int cl = 0; cl < cig; ++cl) {
        const int ci = g * cig + cl;
        device const float* xr = x + (t * cin + ci) * fin;
        device const float* wk = w + (ci * cog + col) * kf;
        for (int i = 0; i < fin; ++i) {
            for (int k = 0; k < kf; ++k) {
                if (i * fs - pad + k == o) { acc += xr[i] * wk[k]; }
            }
        }
    }
    if (act != 0) { acc = bn_act(acc, scale[co], bias[co], act); }
    y[(t * cout + co) * fout + o] = acc;
}

// p = [cout, fout, rows, act]
kernel void dfn_pointwise(
    device const float* y [[buffer(0)]], device const float* pw [[buffer(1)]],
    device const float* scale [[buffer(2)]], device const float* bias [[buffer(3)]],
    device float* z [[buffer(4)]], constant int* p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int cout = p[0], fout = p[1], rows = p[2], act = p[3];
    const int t = int(gid.y), idx = int(gid.x);
    if (t >= rows || idx >= cout * fout) { return; }
    const int co = idx / fout, fo = idx % fout;
    device const float* wr = pw + co * cout;
    device const float* yt = y + t * cout * fout + fo;
    float acc = 0.0f;
    for (int ci = 0; ci < cout; ++ci) { acc += wr[ci] * yt[ci * fout]; }
    z[(t * cout + co) * fout + fo] = bn_act(acc, scale[co], bias[co], act);
}

// p = [ig, hg, outs, in_stride, out_stride, rows, relu, bias]
kernel void dfn_linear(
    device const float* x [[buffer(0)]], device const float* w [[buffer(1)]],
    device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
    constant int* p [[buffer(4)]], uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int ig = p[0], hg = p[1], outs = p[2], rows = p[5];
    const int t = int(gid.y), row = int(gid.x);
    if (t >= rows || row >= outs) { return; }
    float v = dot8(x + t * p[3] + (row / hg) * ig, w + row * ig, 1, ig);
    if (p[7] != 0) { v = v + b[row]; }
    if (p[6] != 0) { v = v > 0.0f ? v : 0.0f; }
    y[t * p[4] + row] = v;
}

// [c][f] -> [f][c]. p = [c, f, rows]
kernel void dfn_to_fc(
    device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
    constant int* p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]
) {
    const int c = p[0], f = p[1], rows = p[2];
    const int t = int(gid.y), idx = int(gid.x);
    if (t >= rows || idx >= c * f) { return; }
    const int ci = idx / f, fi = idx % f;
    y[t * c * f + fi * c + ci] = x[t * c * f + idx];
}

// a += b (mode 0) ou a += max(b, 0) (mode 1). p = [n, rows, mode]
kernel void dfn_add(
    device float* a [[buffer(0)]], device const float* b [[buffer(1)]],
    constant int* p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int n = p[0], rows = p[1];
    const int t = int(gid.y), i = int(gid.x);
    if (t >= rows || i >= n) { return; }
    float v = b[t * n + i];
    if (p[2] == 1) { v = v > 0.0f ? v : 0.0f; }
    a[t * n + i] = a[t * n + i] + v;
}

// c[f][o] = tanh(c[f][o]) + path[o][f]. p = [nb_df, out_ch, rows]
kernel void dfn_coefs(
    device float* c [[buffer(0)]], device const float* path [[buffer(1)]],
    constant int* p [[buffer(2)]], uint2 gid [[thread_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    const int nf = p[0], oc = p[1], rows = p[2];
    const int t = int(gid.y), idx = int(gid.x);
    if (t >= rows || idx >= nf * oc) { return; }
    const int f = idx / oc, o = idx % oc;
    c[t * nf * oc + idx] = dfx_tanh(c[t * nf * oc + idx]) + path[t * nf * oc + o * nf + f];
}

// Boucle temporelle complète d'un GRU (n = 256) par threadgroup de 3n threads ;
// le threadgroup s traite la séquence s. gi inclut b_ih ; whh transposé [n][3n].
// p = [rows]
kernel void dfn_gru(
    device const float* gi0 [[buffer(0)]], device const float* w0 [[buffer(1)]],
    device const float* b0 [[buffer(2)]], device float* hs0 [[buffer(3)]],
    device float* st0 [[buffer(4)]],
    device const float* gi1 [[buffer(5)]], device const float* w1 [[buffer(6)]],
    device const float* b1 [[buffer(7)]], device float* hs1 [[buffer(8)]],
    device float* st1 [[buffer(9)]],
    constant int* p [[buffer(10)]],
    uint r [[thread_index_in_threadgroup]], uint s [[threadgroup_position_in_grid]]
) {
    #pragma METAL fp contract(off)
    constexpr int n = 256;
    threadgroup float h[n];
    threadgroup float gh[3 * n];
    device const float* gi = s == 0 ? gi0 : gi1;
    device const float* whh = s == 0 ? w0 : w1;
    device const float* bhh = s == 0 ? b0 : b1;
    device float* hs = s == 0 ? hs0 : hs1;
    device float* st = s == 0 ? st0 : st1;
    const int j = int(r);
    if (j < n) { h[j] = st[j]; }
    const float bias = bhh[j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int t = 0; t < p[0]; ++t) {
        gh[j] = dot8(h, whh + j, 3 * n, n) + bias;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (j < n) {
            device const float* g = gi + t * 3 * n;
            const float rg = dfx_sigmoid(g[j] + gh[j]);
            const float z = dfx_sigmoid(g[n + j] + gh[n + j]);
            const float c = dfx_tanh(g[2 * n + j] + rg * gh[2 * n + j]);
            const float hn = (1.0f - z) * c + z * h[j];
            h[j] = hn;
            hs[t * n + j] = hn;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (j < n) { st[j] = h[j]; }
}
";

struct Pipelines {
    conv: ComputePipelineState,
    convt: ComputePipelineState,
    pointwise: ComputePipelineState,
    linear: ComputePipelineState,
    to_fc: ComputePipelineState,
    add: ComputePipelineState,
    coefs: ComputePipelineState,
    gru: ComputePipelineState,
}

fn upload(device: &Device, data: &[f32]) -> Result<Buffer> {
    let buffer = device.new_buffer(
        (data.len().max(1) * 4) as u64,
        MTLResourceOptions::StorageModeShared,
    );
    write_f32_buffer(&buffer, data)?;
    Ok(buffer)
}

/// Activation `[lignes][width]` ; `row0` décale toutes les liaisons d'un encodage.
struct Buf {
    buffer: Buffer,
    width: usize,
}

struct GConv<'a> {
    src: &'a Conv,
    weight: Buffer,
    pointwise: Option<Buffer>,
    scale: Buffer,
    bias: Buffer,
}

struct GLin {
    weight: Buffer,
    bias: Option<Buffer>,
    ig: usize,
    hg: usize,
    outs: usize,
}

struct GGru {
    ih: GLin,
    w_hh_t: Buffer,
    b_hh: Buffer,
}

struct GSqueezed {
    lin_in: GLin,
    layers: Vec<GGru>,
    lin_out: Option<GLin>,
}

struct Ctx<'a> {
    device: Device,
    queue: CommandQueue,
    p: Pipelines,
    erb_conv: Vec<GConv<'a>>,
    df_conv0: GConv<'a>,
    df_conv1: GConv<'a>,
    df_fc_emb: GLin,
    enc: GSqueezed,
    erb: GSqueezed,
    df: GSqueezed,
    convp: Vec<GConv<'a>>,
    convt: Vec<GConv<'a>>,
    conv0_out: GConv<'a>,
    df_convp: GConv<'a>,
    df_skip: GLin,
    df_out: GLin,
}

impl<'a> Ctx<'a> {
    fn new(net: &'a Network) -> Result<Self> {
        let device =
            Device::system_default().ok_or_else(|| InferError::Metal("aucun GPU Metal".into()))?;
        let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\n{}\n{KERNELS}",
            super::exact_math::MSL
        );
        let library = device
            .new_library_with_source(&source, &crate::metal_backend::compile_options(false))
            .map_err(|e| InferError::Metal(format!("noyaux DFN3: {e}")))?;
        let pipeline = |name: &str| -> Result<ComputePipelineState> {
            let function = library
                .get_function(name, None)
                .map_err(|e| InferError::Metal(format!("{name}: {e}")))?;
            device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| InferError::Metal(format!("{name}: {e}")))
        };
        let p = Pipelines {
            conv: pipeline("dfn_conv")?,
            convt: pipeline("dfn_convt")?,
            pointwise: pipeline("dfn_pointwise")?,
            linear: pipeline("dfn_linear")?,
            to_fc: pipeline("dfn_to_fc")?,
            add: pipeline("dfn_add")?,
            coefs: pipeline("dfn_coefs")?,
            gru: pipeline("dfn_gru")?,
        };
        if p.gru.max_total_threads_per_threadgroup() < GATES as u64 {
            return Err(InferError::Metal(format!(
                "dfn_gru: {} threads par threadgroup, {GATES} requis",
                p.gru.max_total_threads_per_threadgroup()
            )));
        }
        let conv = |c: &'a Conv| -> Result<GConv<'a>> {
            Ok(GConv {
                src: c,
                weight: upload(&device, &c.weight)?,
                pointwise: (!c.pointwise.is_empty())
                    .then(|| upload(&device, &c.pointwise))
                    .transpose()?,
                scale: upload(&device, &c.scale)?,
                bias: upload(&device, &c.bias)?,
            })
        };
        let lin = |l: &GroupedLinear| -> Result<GLin> {
            Ok(GLin {
                weight: upload(&device, &l.weight)?,
                bias: None,
                ig: l.ig,
                hg: l.hg,
                outs: l.weight.len() / l.ig,
            })
        };
        let gru = |g: &Gru| -> Result<GGru> {
            let n = g.hidden;
            let mut t = vec![0.0; g.w_hh.len()];
            for row in 0..3 * n {
                for k in 0..n {
                    t[k * 3 * n + row] = g.w_hh[row * n + k];
                }
            }
            Ok(GGru {
                ih: GLin {
                    weight: upload(&device, &g.w_ih)?,
                    bias: Some(upload(&device, &g.b_ih)?),
                    ig: g.w_ih.len() / (3 * n),
                    hg: 3 * n,
                    outs: 3 * n,
                },
                w_hh_t: upload(&device, &t)?,
                b_hh: upload(&device, &g.b_hh)?,
            })
        };
        let squeezed = |s: &SqueezedGru| -> Result<GSqueezed> {
            if s.layers.iter().any(|l| l.hidden != HIDDEN) {
                return Err(InferError::Shape("GRU DFN3 de largeur ≠ 256".into()));
            }
            Ok(GSqueezed {
                lin_in: lin(&s.lin_in)?,
                layers: s.layers.iter().map(&gru).collect::<Result<_>>()?,
                lin_out: s.lin_out.as_ref().map(&lin).transpose()?,
            })
        };
        Ok(Self {
            erb_conv: net.erb_conv.iter().map(&conv).collect::<Result<_>>()?,
            df_conv0: conv(&net.df_conv0)?,
            df_conv1: conv(&net.df_conv1)?,
            df_fc_emb: lin(&net.df_fc_emb)?,
            enc: squeezed(&net.enc_gru)?,
            erb: squeezed(&net.erb_gru)?,
            df: squeezed(&net.df_gru)?,
            convp: net.convp.iter().map(&conv).collect::<Result<_>>()?,
            convt: net.convt.iter().map(&conv).collect::<Result<_>>()?,
            conv0_out: conv(&net.conv0_out)?,
            df_convp: conv(&net.df_convp)?,
            df_skip: lin(&net.df_skip)?,
            df_out: lin(&net.df_out)?,
            queue: device.new_command_queue(),
            p,
            device,
        })
    }

    fn act(&self, rows: usize, width: usize) -> Buf {
        Buf {
            buffer: self.device.new_buffer(
                (rows * width * 4) as u64,
                MTLResourceOptions::StorageModeShared,
            ),
            width,
        }
    }
}

/// Encodeur à décalage de lignes : toutes les activations sont liées à partir de `row0`.
struct Enc<'e> {
    enc: &'e ComputeCommandEncoderRef,
    row0: usize,
    rows: usize,
}

impl Enc<'_> {
    fn bind(&self, index: u64, a: &Buf) {
        self.enc
            .set_buffer(index, Some(&a.buffer), (self.row0 * a.width * 4) as u64);
    }

    fn params(&self, index: u64, p: &[i32]) {
        self.enc
            .set_bytes(index, std::mem::size_of_val(p) as u64, p.as_ptr().cast());
    }

    fn grid(&self, pso: &ComputePipelineState, width: usize) {
        self.enc.set_compute_pipeline_state(pso);
        self.enc.dispatch_threads(
            MTLSize::new(width as u64, self.rows as u64, 1),
            MTLSize::new(width.min(64) as u64, 1, 1),
        );
    }

    /// `Conv::forward` sur toutes les lignes ; `tmp` reçoit la conv avant la 1×1.
    #[allow(clippy::too_many_arguments)]
    fn conv(&self, p: &Pipelines, c: &GConv, x: &Buf, tmp: &Buf, y: &Buf, fin: usize, zb: i32) {
        let s = c.src;
        let fout = s.fout(fin);
        let act = match (&c.pointwise, s.act) {
            (Some(_), _) => 0,
            (None, Act::Relu) => 1,
            (None, Act::Sigmoid) => 2,
        };
        let dims = [
            s.cin, s.cout, s.groups, s.kt, s.kf, s.fstride, fin, fout, self.rows,
        ]
        .map(|v| v as i32);
        let params = [dims.as_slice(), &[zb, act][..]].concat();
        self.bind(0, x);
        self.enc.set_buffer(1, Some(&c.weight), 0);
        self.enc.set_buffer(2, Some(&c.scale), 0);
        self.enc.set_buffer(3, Some(&c.bias), 0);
        self.bind(4, if c.pointwise.is_some() { tmp } else { y });
        self.params(5, &params);
        let pso = if s.transposed { &p.convt } else { &p.conv };
        self.grid(pso, s.cout * fout);
        if let Some(pw) = &c.pointwise {
            let fin_act = i32::from(s.act == Act::Sigmoid) + 1;
            self.bind(0, tmp);
            self.enc.set_buffer(1, Some(pw), 0);
            self.bind(4, y);
            self.params(5, &[s.cout as i32, fout as i32, self.rows as i32, fin_act]);
            self.grid(&p.pointwise, s.cout * fout);
        }
    }

    fn linear(&self, p: &Pipelines, l: &GLin, x: &Buf, y: &Buf, relu: bool) {
        self.bind(0, x);
        self.enc.set_buffer(1, Some(&l.weight), 0);
        self.enc.set_buffer(2, l.bias.as_deref(), 0);
        self.bind(3, y);
        self.params(
            4,
            &[
                l.ig as i32,
                l.hg as i32,
                l.outs as i32,
                x.width as i32,
                y.width as i32,
                self.rows as i32,
                i32::from(relu),
                i32::from(l.bias.is_some()),
            ],
        );
        self.grid(&p.linear, l.outs);
    }

    fn to_fc(&self, p: &Pipelines, x: &Buf, y: &Buf, c: usize, f: usize) {
        self.bind(0, x);
        self.bind(1, y);
        self.params(2, &[c as i32, f as i32, self.rows as i32]);
        self.grid(&p.to_fc, c * f);
    }

    fn add(&self, p: &Pipelines, a: &Buf, b: &Buf, relu: bool) {
        self.bind(0, a);
        self.bind(1, b);
        self.params(2, &[a.width as i32, self.rows as i32, i32::from(relu)]);
        self.grid(&p.add, a.width);
    }

    /// Une ou deux séquences GRU en parallèle, une par threadgroup.
    fn gru(&self, p: &Pipelines, seqs: &[(&GGru, &Buf, &Buf, &Buffer)]) {
        self.enc.set_compute_pipeline_state(&p.gru);
        for slot in 0..2 {
            let (g, gi, hs, st) = seqs[slot.min(seqs.len() - 1)];
            let base = 5 * slot as u64;
            self.bind(base, gi);
            self.enc.set_buffer(base + 1, Some(&g.w_hh_t), 0);
            self.enc.set_buffer(base + 2, Some(&g.b_hh), 0);
            self.bind(base + 3, hs);
            self.enc.set_buffer(base + 4, Some(st), 0);
        }
        self.params(10, &[self.rows as i32]);
        self.enc.dispatch_thread_groups(
            MTLSize::new(seqs.len() as u64, 1, 1),
            MTLSize::new(GATES as u64, 1, 1),
        );
    }
}

/// Sorties du réseau pour les `erb.len() / NB_ERB` pas : masques et coefficients DF.
pub(super) fn run(net: &Network, erb: &[f32], cplx: &[f32]) -> Result<(Vec<f32>, Vec<f32>)> {
    let steps = erb.len() / NB_ERB;
    if cplx.len() != steps * 2 * NB_DF {
        return Err(InferError::Dimension("features DFN3 incohérentes".into()));
    }
    let ctx = Ctx::new(net)?;
    let c = &ctx;
    let rows = CHUNK + PREFIX;
    let a = |w| c.act(rows, w);
    let (erbx, specx) = (a(NB_ERB), a(2 * NB_DF));
    let (e0, e1, e2, e3) = (a(CH * 32), a(CH * 16), a(CH * 8), a(CH * 8));
    let (c0, c1) = (a(CH * NB_DF), a(CH * NB_DF / 2));
    let tmp = a(CH * NB_DF);
    let (cemb_in, cemb, emb_in) = (a(CH * NB_DF / 2), a(EMB), a(EMB));
    let (h_in, h_in2, gi, gi2) = (a(HIDDEN), a(HIDDEN), a(GATES), a(GATES));
    let (hs, hs2, hs3) = (a(HIDDEN), a(HIDDEN), a(HIDDEN));
    let (emb, d, skip) = (a(EMB), a(EMB), a(HIDDEN));
    let (x8, t8, x16, t16, x32, t32) = (
        a(CH * 8),
        a(CH * 8),
        a(CH * 16),
        a(CH * 16),
        a(CH * 32),
        a(CH * 32),
    );
    let mask = a(NB_ERB);
    let (coefs, path) = (a(NB_DF * DF_OUT_CH), a(NB_DF * DF_OUT_CH));
    let state = || upload(&c.device, &[0.0; HIDDEN]);
    let (st_enc, st_e0, st_e1, st_d0, st_d1) = (state()?, state()?, state()?, state()?, state()?);

    let mut out_mask = Vec::with_capacity(steps * NB_ERB);
    let mut out_coefs = Vec::with_capacity(steps * NB_DF * DF_OUT_CH);
    let mut start = 0;
    while start < steps {
        let tc = CHUNK.min(steps - start);
        // Ligne locale j ↔ pas global start − PREFIX + j ; pas négatifs nuls.
        let fill = |src: &[f32], width: usize| {
            let mut v = vec![0.0_f32; (tc + PREFIX) * width];
            let lo = start.saturating_sub(PREFIX);
            let off = PREFIX - (start - lo);
            v[off * width..].copy_from_slice(&src[lo * width..(start + tc) * width]);
            v
        };
        write_f32_buffer(&erbx.buffer, &fill(erb, NB_ERB))?;
        write_f32_buffer(&specx.buffer, &fill(cplx, 2 * NB_DF))?;
        let cb = c.queue.new_command_buffer();
        let raw = cb.new_compute_command_encoder();
        let p = &c.p;
        // Encodeur : toutes les lignes, contexte compris.
        let all = Enc {
            enc: raw,
            row0: 0,
            rows: tc + PREFIX,
        };
        all.conv(p, &c.erb_conv[0], &erbx, &tmp, &e0, NB_ERB, 0);
        all.conv(p, &c.erb_conv[1], &e0, &tmp, &e1, NB_ERB, 0);
        all.conv(p, &c.erb_conv[2], &e1, &tmp, &e2, NB_ERB / 2, 0);
        all.conv(p, &c.erb_conv[3], &e2, &tmp, &e3, NB_ERB / 4, 0);
        all.conv(p, &c.df_conv0, &specx, &tmp, &c0, NB_DF, 0);
        all.conv(p, &c.df_conv1, &c0, &tmp, &c1, NB_DF, 0);
        // Pas utiles seulement.
        let e = Enc {
            enc: raw,
            row0: PREFIX,
            rows: tc,
        };
        e.to_fc(p, &c1, &cemb_in, CH, NB_DF / 2);
        e.linear(p, &c.df_fc_emb, &cemb_in, &cemb, false);
        e.to_fc(p, &e3, &emb_in, CH, NB_ERB / 4);
        e.add(p, &emb_in, &cemb, true);
        e.linear(p, &c.enc.lin_in, &emb_in, &h_in, true);
        e.linear(p, &c.enc.layers[0].ih, &h_in, &gi, false);
        e.gru(p, &[(&c.enc.layers[0], &gi, &hs, &st_enc)]);
        if let Some(lo) = &c.enc.lin_out {
            e.linear(p, lo, &hs, &emb, true);
        }
        // Décodeurs ERB et DF : leurs GRU tournent côte à côte.
        e.linear(p, &c.erb.lin_in, &emb, &h_in, true);
        e.linear(p, &c.df.lin_in, &emb, &h_in2, true);
        e.linear(p, &c.erb.layers[0].ih, &h_in, &gi, false);
        e.linear(p, &c.df.layers[0].ih, &h_in2, &gi2, false);
        e.gru(
            p,
            &[
                (&c.erb.layers[0], &gi, &hs, &st_e0),
                (&c.df.layers[0], &gi2, &hs2, &st_d0),
            ],
        );
        e.linear(p, &c.erb.layers[1].ih, &hs, &gi, false);
        e.linear(p, &c.df.layers[1].ih, &hs2, &gi2, false);
        e.gru(
            p,
            &[
                (&c.erb.layers[1], &gi, &hs, &st_e1),
                (&c.df.layers[1], &gi2, &hs3, &st_d1),
            ],
        );
        if let Some(lo) = &c.erb.lin_out {
            e.linear(p, lo, &hs, &d, true);
        }
        // from_fc(d, 64, 8) = to_fc(d, 8, 64).
        e.to_fc(p, &d, &x8, NB_ERB / 4, CH);
        e.conv(p, &c.convp[3], &e3, &tmp, &t8, NB_ERB / 4, 0);
        e.add(p, &t8, &x8, false);
        e.conv(p, &c.convt[2], &t8, &tmp, &x8, NB_ERB / 4, 0);
        e.conv(p, &c.convp[2], &e2, &tmp, &t8, NB_ERB / 4, 0);
        e.add(p, &t8, &x8, false);
        e.conv(p, &c.convt[1], &t8, &tmp, &x16, NB_ERB / 4, 0);
        e.conv(p, &c.convp[1], &e1, &tmp, &t16, NB_ERB / 2, 0);
        e.add(p, &t16, &x16, false);
        e.conv(p, &c.convt[0], &t16, &tmp, &x32, NB_ERB / 2, 0);
        e.conv(p, &c.convp[0], &e0, &tmp, &t32, NB_ERB, 0);
        e.add(p, &t32, &x32, false);
        e.conv(p, &c.conv0_out, &t32, &tmp, &mask, NB_ERB, 0);
        // Décodeur DF.
        e.linear(p, &c.df_skip, &emb, &skip, false);
        e.add(p, &hs3, &skip, false);
        e.linear(p, &c.df_out, &hs3, &coefs, false);
        // c0 aux pas globaux négatifs vaut zéro (historique CPU vierge).
        let zb = -(start.min(PREFIX) as i32);
        e.conv(p, &c.df_convp, &c0, &tmp, &path, NB_DF, zb);
        e.bind(0, &coefs);
        e.bind(1, &path);
        e.params(2, &[NB_DF as i32, DF_OUT_CH as i32, tc as i32]);
        e.grid(&p.coefs, NB_DF * DF_OUT_CH);
        raw.end_encoding();
        commit_and_wait(cb)?;
        out_mask.extend(read_f32_buffer_at(
            &mask.buffer,
            PREFIX * NB_ERB,
            tc * NB_ERB,
        )?);
        let width = NB_DF * DF_OUT_CH;
        out_coefs.extend(read_f32_buffer_at(
            &coefs.buffer,
            PREFIX * width,
            tc * width,
        )?);
        start += tc;
    }
    Ok((out_mask, out_coefs))
}

/// Sonde : µs GPU par pas de la boucle GRU persistante, `seqs` séquences côte à côte.
#[cfg(test)]
pub(super) fn gru_probe(net: &Network, steps: usize, seqs: usize) -> Result<f64> {
    let ctx = Ctx::new(net)?;
    let gi = ctx.act(steps, GATES);
    let pattern: Vec<f32> = (0..steps * GATES)
        .map(|i| ((i * 7919 % 997) as f32 - 498.0) / 997.0)
        .collect();
    write_f32_buffer(&gi.buffer, &pattern)?;
    let (hs, hs2) = (ctx.act(steps, HIDDEN), ctx.act(steps, HIDDEN));
    let (st, st2) = (
        upload(&ctx.device, &[0.0; HIDDEN])?,
        upload(&ctx.device, &[0.0; HIDDEN])?,
    );
    let layer = &ctx.erb.layers[0];
    let all = [(layer, &gi, &hs, &st), (layer, &gi, &hs2, &st2)];
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let cb = ctx.queue.new_command_buffer();
        let raw = cb.new_compute_command_encoder();
        Enc {
            enc: raw,
            row0: 0,
            rows: steps,
        }
        .gru(&ctx.p, &all[..seqs]);
        raw.end_encoding();
        commit_and_wait(cb)?;
        let ns = crate::metal_backend::command_buffer_gpu_ns(cb) as f64;
        best = best.min(ns / 1e3 / steps as f64);
    }
    Ok(best)
}
