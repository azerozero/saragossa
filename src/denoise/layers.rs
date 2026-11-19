//! Couches CPU f32 de DeepFilterNet3, exécutées une trame à la fois.
//!
//! Toutes les convolutions du modèle sont causales en temps (stride temporel 1)
//! : une trame de sortie dépend des `kt` dernières trames d'entrée, conservées
//! par l'appelant. Les activations d'une trame sont rangées `[canal][fréquence]`,
//! comme une tranche `[C, F]` du tenseur PyTorch `[B, C, T, F]`.

use std::collections::HashMap;

use crate::{InferError, Result, Tensor};

/// Produit scalaire à huit accumulateurs, que LLVM vectorise en NEON.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0.0_f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut tail = 0.0;
    for (x, y) in ra.iter().zip(rb) {
        tail += x * y;
    }
    acc.iter().sum::<f32>() + tail
}

#[inline]
pub(crate) fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Fonctions transcendantes : libm (défaut historique) ou variante commune au GPU.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Math {
    #[default]
    Libm,
    /// [`super::exact_math`], reproduite au bit près par le chemin Metal.
    Common,
}

impl Math {
    #[inline]
    pub(crate) fn sigmoid(self, x: f32) -> f32 {
        match self {
            Self::Libm => sigmoid(x),
            Self::Common => super::exact_math::sigmoid(x),
        }
    }

    #[inline]
    pub(crate) fn tanh(self, x: f32) -> f32 {
        match self {
            Self::Libm => x.tanh(),
            Self::Common => super::exact_math::tanh(x),
        }
    }
}

/// Activation de sortie d'un bloc `Conv2dNormAct`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Act {
    Relu,
    Sigmoid,
}

pub(crate) fn take(w: &mut HashMap<String, Tensor>, name: &str) -> Result<Tensor> {
    w.remove(name)
        .ok_or_else(|| InferError::MissingWeight(name.to_string()))
}

pub(crate) fn take_shape(
    w: &mut HashMap<String, Tensor>,
    name: &str,
    shape: &[usize],
) -> Result<Vec<f32>> {
    let t = take(w, name)?;
    if t.shape() != shape {
        return Err(InferError::Shape(format!(
            "{name}: attendu {shape:?}, trouvé {:?}",
            t.shape()
        )));
    }
    Ok(t.into_data())
}

/// Description statique d'un bloc `Conv2dNormAct` / `ConvTranspose2dNormAct`.
pub(crate) struct ConvSpec {
    pub prefix: &'static str,
    pub cin: usize,
    pub cout: usize,
    pub groups: usize,
    pub kt: usize,
    pub kf: usize,
    pub fstride: usize,
    pub transposed: bool,
    /// Index Sequential de la convolution principale (1 si un `ConstantPad2d` précède).
    pub conv_idx: usize,
    /// Une convolution 1×1 suit la convolution groupée (`separable`).
    pub pointwise: bool,
    pub act: Act,
}

/// Bloc conv + 1×1 éventuelle + BatchNorm (repliée) + activation.
pub(crate) struct Conv {
    pub(super) cin: usize,
    pub(super) cout: usize,
    pub(super) groups: usize,
    pub(super) kt: usize,
    pub(super) kf: usize,
    pub(super) fstride: usize,
    pub(super) transposed: bool,
    /// Conv : `[cout][cin/g][kt][kf]`. Transposée : `[cin][cout/g][kf]` (kt = 1).
    pub(super) weight: Vec<f32>,
    /// 1×1 `[cout][cout]` avec la BatchNorm repliée, sinon vide.
    pub(super) pointwise: Vec<f32>,
    pub(super) scale: Vec<f32>,
    pub(super) bias: Vec<f32>,
    pub(super) act: Act,
}

impl Conv {
    pub(crate) fn load(w: &mut HashMap<String, Tensor>, s: &ConvSpec) -> Result<Self> {
        let p = s.prefix;
        let shape = if s.transposed {
            vec![s.cin, s.cout / s.groups, s.kt, s.kf]
        } else {
            vec![s.cout, s.cin / s.groups, s.kt, s.kf]
        };
        let weight = take_shape(w, &format!("{p}.{}.weight", s.conv_idx), &shape)?;
        let mut idx = s.conv_idx + 1;
        let mut pointwise = Vec::new();
        if s.pointwise {
            pointwise = take_shape(w, &format!("{p}.{idx}.weight"), &[s.cout, s.cout, 1, 1])?;
            idx += 1;
        }
        let bn = |w: &mut HashMap<String, Tensor>, n: &str| {
            take_shape(w, &format!("{p}.{idx}.{n}"), &[s.cout])
        };
        if !w.contains_key(&format!("{p}.{idx}.running_mean")) {
            // BatchNorm déjà repliée dans la convolution (export ONNX) : biais seul.
            return Ok(Self {
                cin: s.cin,
                cout: s.cout,
                groups: s.groups,
                kt: s.kt,
                kf: s.kf,
                fstride: s.fstride,
                transposed: s.transposed,
                weight,
                pointwise,
                scale: vec![1.0; s.cout],
                bias: bn(w, "bias")?,
                act: s.act,
            });
        }
        let (gamma, beta) = (bn(w, "weight")?, bn(w, "bias")?);
        let (mean, var) = (bn(w, "running_mean")?, bn(w, "running_var")?);
        w.remove(&format!("{p}.{idx}.num_batches_tracked"));
        // BatchNorm2d en évaluation, eps PyTorch par défaut 1e-5.
        let scale: Vec<f32> = gamma
            .iter()
            .zip(&var)
            .map(|(g, v)| g / (v + 1e-5).sqrt())
            .collect();
        let bias = beta
            .iter()
            .zip(&mean)
            .zip(&scale)
            .map(|((b, m), s)| b - m * s)
            .collect();
        Ok(Self {
            cin: s.cin,
            cout: s.cout,
            groups: s.groups,
            kt: s.kt,
            kf: s.kf,
            fstride: s.fstride,
            transposed: s.transposed,
            weight,
            pointwise,
            scale,
            bias,
            act: s.act,
        })
    }

    pub(crate) fn kt(&self) -> usize {
        self.kt
    }

    /// Nombre de bins de sortie pour `fin` bins d'entrée.
    pub(crate) fn fout(&self, fin: usize) -> usize {
        let pad = self.kf / 2;
        if self.transposed {
            // padding = kf/2, output_padding = kf/2, dilatation 1.
            (fin - 1) * self.fstride + self.kf - 2 * pad + pad
        } else {
            (fin + 2 * pad - self.kf) / self.fstride + 1
        }
    }

    /// `hist[dt]` = trame `t - kt + 1 + dt`, chacune `[cin][fin]`. Ecrit `[cout][fout]`.
    pub(crate) fn forward(&self, hist: &[&[f32]], fin: usize, out: &mut Vec<f32>, math: Math) {
        let fout = self.fout(fin);
        let mut y = vec![0.0_f32; self.cout * fout];
        let (cig, cog) = (self.cin / self.groups, self.cout / self.groups);
        let pad = (self.kf / 2) as isize;
        if self.transposed {
            let x = hist[0];
            for ci in 0..self.cin {
                let g = ci / cig;
                for col in 0..cog {
                    let co = g * cog + col;
                    let wk = &self.weight[(ci * cog + col) * self.kf..][..self.kf];
                    for i in 0..fin {
                        let xv = x[ci * fin + i];
                        for (k, &wv) in wk.iter().enumerate() {
                            let o = (i * self.fstride) as isize - pad + k as isize;
                            if (0..fout as isize).contains(&o) {
                                y[co * fout + o as usize] += xv * wv;
                            }
                        }
                    }
                }
            }
        } else {
            for co in 0..self.cout {
                let g = co / cog;
                let yrow = &mut y[co * fout..(co + 1) * fout];
                for cl in 0..cig {
                    let ci = g * cig + cl;
                    for (dt, frame) in hist.iter().enumerate() {
                        let x = &frame[ci * fin..(ci + 1) * fin];
                        let wk =
                            &self.weight[((co * cig + cl) * self.kt + dt) * self.kf..][..self.kf];
                        for (fo, yv) in yrow.iter_mut().enumerate() {
                            let base = (fo * self.fstride) as isize - pad;
                            for (k, &wv) in wk.iter().enumerate() {
                                let fi = base + k as isize;
                                if (0..fin as isize).contains(&fi) {
                                    *yv += x[fi as usize] * wv;
                                }
                            }
                        }
                    }
                }
            }
        }
        if !self.pointwise.is_empty() {
            let mut z = vec![0.0_f32; self.cout * fout];
            for co in 0..self.cout {
                let wrow = &self.pointwise[co * self.cout..(co + 1) * self.cout];
                let zrow = &mut z[co * fout..(co + 1) * fout];
                for (ci, &wv) in wrow.iter().enumerate() {
                    for (zv, &xv) in zrow.iter_mut().zip(&y[ci * fout..(ci + 1) * fout]) {
                        *zv += wv * xv;
                    }
                }
            }
            y = z;
        }
        for co in 0..self.cout {
            let (s, b) = (self.scale[co], self.bias[co]);
            for v in &mut y[co * fout..(co + 1) * fout] {
                let a = *v * s + b;
                *v = match self.act {
                    Act::Relu => a.max(0.0),
                    Act::Sigmoid => math.sigmoid(a),
                };
            }
        }
        *out = y;
    }
}

/// `GroupedLinearEinsum` : `y[g·H/G + h] = Σ_i x[g·I/G + i] · W[g][i][h]`.
pub(crate) struct GroupedLinear {
    pub(super) groups: usize,
    pub(super) ig: usize,
    pub(super) hg: usize,
    /// Transposé à `[g][h][i]` pour des produits scalaires contigus.
    pub(super) weight: Vec<f32>,
}

impl GroupedLinear {
    pub(crate) fn load(
        w: &mut HashMap<String, Tensor>,
        name: &str,
        input: usize,
        hidden: usize,
        groups: usize,
    ) -> Result<Self> {
        let (ig, hg) = (input / groups, hidden / groups);
        let raw = take_shape(w, name, &[groups, ig, hg])?;
        let mut weight = vec![0.0; raw.len()];
        for g in 0..groups {
            for i in 0..ig {
                for h in 0..hg {
                    weight[(g * hg + h) * ig + i] = raw[(g * ig + i) * hg + h];
                }
            }
        }
        Ok(Self {
            groups,
            ig,
            hg,
            weight,
        })
    }

    pub(crate) fn forward(&self, x: &[f32], y: &mut [f32]) {
        for g in 0..self.groups {
            let xg = &x[g * self.ig..(g + 1) * self.ig];
            for h in 0..self.hg {
                let row = g * self.hg + h;
                y[row] = dot(xg, &self.weight[row * self.ig..(row + 1) * self.ig]);
            }
        }
    }
}

/// Couche `nn.GRU` PyTorch (portes r, z, n), un pas à la fois.
pub(crate) struct Gru {
    pub(super) hidden: usize,
    pub(super) w_ih: Vec<f32>,
    pub(super) w_hh: Vec<f32>,
    pub(super) b_ih: Vec<f32>,
    pub(super) b_hh: Vec<f32>,
}

impl Gru {
    pub(crate) fn load(
        w: &mut HashMap<String, Tensor>,
        prefix: &str,
        layer: usize,
        input: usize,
        hidden: usize,
    ) -> Result<Self> {
        let name = |k: &str| format!("{prefix}.{k}_l{layer}");
        Ok(Self {
            hidden,
            w_ih: take_shape(w, &name("weight_ih"), &[3 * hidden, input])?,
            w_hh: take_shape(w, &name("weight_hh"), &[3 * hidden, hidden])?,
            b_ih: take_shape(w, &name("bias_ih"), &[3 * hidden])?,
            b_hh: take_shape(w, &name("bias_hh"), &[3 * hidden])?,
        })
    }

    /// Met à jour `h` en place à partir de l'entrée `x`.
    pub(crate) fn step(&self, x: &[f32], h: &mut [f32], math: Math) {
        self.recur(&self.input_gates(x), h, math);
    }

    /// Portes d'entrée `W_ih · x + b_ih`, indépendantes de l'état : calculables d'avance.
    pub(crate) fn input_gates(&self, x: &[f32]) -> Vec<f32> {
        let ni = x.len();
        (0..3 * self.hidden)
            .map(|r| dot(x, &self.w_ih[r * ni..(r + 1) * ni]) + self.b_ih[r])
            .collect()
    }

    /// Partie récurrente du pas, à portes d'entrée `gi` déjà calculées.
    pub(crate) fn recur(&self, gi: &[f32], h: &mut [f32], math: Math) {
        let n = self.hidden;
        let gh: Vec<f32> = (0..3 * n)
            .map(|r| dot(h, &self.w_hh[r * n..(r + 1) * n]) + self.b_hh[r])
            .collect();
        for j in 0..n {
            let r = math.sigmoid(gi[j] + gh[j]);
            let z = math.sigmoid(gi[n + j] + gh[n + j]);
            let c = math.tanh(gi[2 * n + j] + r * gh[2 * n + j]);
            h[j] = (1.0 - z) * c + z * h[j];
        }
    }
}

/// `nn.Linear(n, 1)`.
pub(crate) struct Linear1 {
    pub(super) weight: Vec<f32>,
    pub(super) bias: f32,
}

impl Linear1 {
    pub(crate) fn load(
        w: &mut HashMap<String, Tensor>,
        prefix: &str,
        input: usize,
    ) -> Result<Self> {
        let weight = take_shape(w, &format!("{prefix}.weight"), &[1, input])?;
        let bias = take_shape(w, &format!("{prefix}.bias"), &[1])?;
        Ok(Self {
            weight,
            bias: bias[0],
        })
    }

    pub(crate) fn forward(&self, x: &[f32]) -> f32 {
        dot(x, &self.weight) + self.bias
    }
}
