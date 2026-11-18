//! Graphe DeepFilterNet3 (`df/deepfilternet3.py`) : encodeur, décodeur ERB et décodeur DF.

use std::collections::HashMap;

use super::dsp::{NB_DF, NB_ERB};
use super::layers::{sigmoid, take_shape, Act, Conv, ConvSpec, GroupedLinear, Gru, Linear1};
use crate::{InferError, Result, Tensor};

mod batch;

const CH: usize = 64;
const EMB: usize = CH * NB_ERB / 4; // 512
const HIDDEN: usize = 256;
const DF_ORDER: usize = 5;
const DF_OUT_CH: usize = 2 * DF_ORDER;
const LSNR_MIN: f32 = -15.0;
const LSNR_MAX: f32 = 35.0;

#[allow(clippy::too_many_arguments)]
const fn spec(
    prefix: &'static str,
    cin: usize,
    cout: usize,
    groups: usize,
    kt: usize,
    kf: usize,
    fstride: usize,
    transposed: bool,
    pointwise: bool,
    act: Act,
) -> ConvSpec {
    ConvSpec {
        prefix,
        cin,
        cout,
        groups,
        kt,
        kf,
        fstride,
        transposed,
        // Un `ConstantPad2d` causal précède la conv dès que kt > 1.
        conv_idx: if kt > 1 { 1 } else { 0 },
        pointwise,
        act,
    }
}

/// Une couche récurrente `SqueezedGRU_S` : linéaire groupée, GRU, linéaire groupée optionnelle.
struct SqueezedGru {
    lin_in: GroupedLinear,
    layers: Vec<Gru>,
    lin_out: Option<GroupedLinear>,
}

impl SqueezedGru {
    fn load(
        w: &mut HashMap<String, Tensor>,
        prefix: &str,
        num_layers: usize,
        groups: usize,
        output: Option<usize>,
    ) -> Result<Self> {
        let lin_in = GroupedLinear::load(
            w,
            &format!("{prefix}.linear_in.0.weight"),
            EMB,
            HIDDEN,
            groups,
        )?;
        let layers = (0..num_layers)
            .map(|l| Gru::load(w, &format!("{prefix}.gru"), l, HIDDEN, HIDDEN))
            .collect::<Result<Vec<_>>>()?;
        let lin_out = output
            .map(|o| {
                GroupedLinear::load(
                    w,
                    &format!("{prefix}.linear_out.0.weight"),
                    HIDDEN,
                    o,
                    groups,
                )
            })
            .transpose()?;
        Ok(Self {
            lin_in,
            layers,
            lin_out,
        })
    }

    fn step(&self, x: &[f32], h: &mut [Vec<f32>], out: &mut Vec<f32>) {
        self.recur(&self.input_gates(x), h, out);
    }

    /// `linear_in`, ReLU puis portes d'entrée de la première couche : sans état.
    fn input_gates(&self, x: &[f32]) -> Vec<f32> {
        let mut a = vec![0.0; HIDDEN];
        self.lin_in.forward(x, &mut a);
        a.iter_mut().for_each(|v| *v = v.max(0.0));
        self.layers[0].input_gates(&a)
    }

    /// Reste du pas, récurrent, à partir des portes de [`Self::input_gates`].
    fn recur(&self, gi0: &[f32], h: &mut [Vec<f32>], out: &mut Vec<f32>) {
        let mut a = vec![0.0; HIDDEN];
        for (l, (layer, state)) in self.layers.iter().zip(h.iter_mut()).enumerate() {
            if l == 0 {
                layer.recur(gi0, state);
            } else {
                layer.step(&a, state);
            }
            a.copy_from_slice(state);
        }
        match &self.lin_out {
            Some(lin) => {
                out.resize(EMB, 0.0);
                lin.forward(&a, out);
                out.iter_mut().for_each(|v| *v = v.max(0.0));
            }
            None => *out = a,
        }
    }
}

/// Poids DeepFilterNet3 chargés et repliés pour l'inférence CPU.
pub struct Network {
    erb_conv: [Conv; 4],
    df_conv0: Conv,
    df_conv1: Conv,
    df_fc_emb: GroupedLinear,
    enc_gru: SqueezedGru,
    lsnr_fc: Linear1,
    erb_gru: SqueezedGru,
    convp: [Conv; 4],
    convt: [Conv; 3],
    conv0_out: Conv,
    df_convp: Conv,
    df_gru: SqueezedGru,
    df_skip: GroupedLinear,
    df_out: GroupedLinear,
    /// `[NB_ERB][481]`, matrice inverse 0/1 du masque ERB.
    pub(crate) erb_inv: Vec<f32>,
    /// `[481][NB_ERB]`, banque ERB normalisée (sert à vérifier les largeurs).
    pub(crate) erb_fb: Vec<f32>,
}

/// État récurrent et historiques temporels d'un flux.
pub(crate) struct NetState {
    erb_hist: Vec<Vec<f32>>,
    spec_hist: Vec<Vec<f32>>,
    c0_hist: Vec<Vec<f32>>,
    h_enc: Vec<Vec<f32>>,
    h_erb: Vec<Vec<f32>>,
    h_df: Vec<Vec<f32>>,
}

/// Sorties d'un pas du réseau.
pub(crate) struct StepOut {
    /// Gains ERB `[NB_ERB]`.
    pub mask: Vec<f32>,
    /// Coefficients DF `[NB_DF][DF_ORDER][2]`.
    pub coefs: Vec<f32>,
    pub lsnr: f32,
}

impl Network {
    pub(crate) fn load(mut w: HashMap<String, Tensor>, freq_bins: usize) -> Result<Self> {
        use Act::{Relu, Sigmoid};
        let w = &mut w;
        let erb_conv = [
            Conv::load(
                w,
                &spec("enc.erb_conv0", 1, CH, 1, 3, 3, 1, false, false, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv1", CH, CH, CH, 1, 3, 2, false, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv2", CH, CH, CH, 1, 3, 2, false, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv3", CH, CH, CH, 1, 3, 1, false, true, Relu),
            )?,
        ];
        let df_conv0 = Conv::load(
            w,
            &spec("enc.df_conv0", 2, CH, 2, 3, 3, 1, false, true, Relu),
        )?;
        let df_conv1 = Conv::load(
            w,
            &spec("enc.df_conv1", CH, CH, CH, 1, 3, 2, false, true, Relu),
        )?;
        let df_fc_emb = GroupedLinear::load(w, "enc.df_fc_emb.0.weight", CH * NB_DF / 2, EMB, 32)?;
        let enc_gru = SqueezedGru::load(w, "enc.emb_gru", 1, 16, Some(EMB))?;
        let lsnr_fc = Linear1::load(w, "enc.lsnr_fc.0", EMB)?;
        let erb_gru = SqueezedGru::load(w, "erb_dec.emb_gru", 2, 16, Some(EMB))?;
        let p = |name| spec(name, CH, CH, CH, 1, 1, 1, false, false, Relu);
        let convp = [
            Conv::load(w, &p("erb_dec.conv0p"))?,
            Conv::load(w, &p("erb_dec.conv1p"))?,
            Conv::load(w, &p("erb_dec.conv2p"))?,
            Conv::load(w, &p("erb_dec.conv3p"))?,
        ];
        let convt = [
            Conv::load(
                w,
                &spec("erb_dec.convt1", CH, CH, CH, 1, 3, 2, true, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("erb_dec.convt2", CH, CH, CH, 1, 3, 2, true, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("erb_dec.convt3", CH, CH, CH, 1, 3, 1, false, true, Relu),
            )?,
        ];
        let conv0_out = Conv::load(
            w,
            &spec(
                "erb_dec.conv0_out",
                CH,
                1,
                1,
                1,
                3,
                1,
                false,
                false,
                Sigmoid,
            ),
        )?;
        let df_convp = Conv::load(
            w,
            &spec(
                "df_dec.df_convp",
                CH,
                DF_OUT_CH,
                2,
                5,
                1,
                1,
                false,
                true,
                Relu,
            ),
        )?;
        let df_gru = SqueezedGru::load(w, "df_dec.df_gru", 2, 8, None)?;
        let df_skip = GroupedLinear::load(w, "df_dec.df_skip.weight", EMB, HIDDEN, 16)?;
        let df_out =
            GroupedLinear::load(w, "df_dec.df_out.0.weight", HIDDEN, NB_DF * DF_OUT_CH, 16)?;
        let erb_inv = take_shape(w, "mask.erb_inv_fb", &[NB_ERB, freq_bins])?;
        let erb_fb = take_shape(w, "erb_fb", &[freq_bins, NB_ERB])?;
        // df_fc_a et la tête de LSNR du décodeur DF ne servent pas à la sortie.
        w.remove("df_dec.df_fc_a.0.weight");
        w.remove("df_dec.df_fc_a.0.bias");
        if let Some(name) = w.keys().next() {
            return Err(InferError::Config(format!(
                "poids DeepFilterNet3 inattendu: {name}"
            )));
        }
        Ok(Self {
            erb_conv,
            df_conv0,
            df_conv1,
            df_fc_emb,
            enc_gru,
            lsnr_fc,
            erb_gru,
            convp,
            convt,
            conv0_out,
            df_convp,
            df_gru,
            df_skip,
            df_out,
            erb_inv,
            erb_fb,
        })
    }

    pub(crate) fn new_state(&self) -> NetState {
        let zeros = |n: usize, len: usize| vec![vec![0.0_f32; len]; n];
        NetState {
            erb_hist: zeros(self.erb_conv[0].kt(), NB_ERB),
            spec_hist: zeros(self.df_conv0.kt(), 2 * NB_DF),
            c0_hist: zeros(self.df_convp.kt(), CH * NB_DF),
            h_enc: zeros(1, HIDDEN),
            h_erb: zeros(2, HIDDEN),
            h_df: zeros(2, HIDDEN),
        }
    }

    /// Un pas du réseau sur les features décalées `s[τ] = feat[τ + conv_lookahead]`.
    pub(crate) fn step(&self, st: &mut NetState, erb: &[f32], cplx: &[f32]) -> StepOut {
        push(&mut st.erb_hist, erb);
        push(&mut st.spec_hist, cplx);
        let enc = self.encode(&hist(&st.erb_hist), &hist(&st.spec_hist));
        let mut emb = Vec::new();
        self.enc_gru.recur(&enc.gates, &mut st.h_enc, &mut emb);
        let (mut d, mut c) = (Vec::new(), Vec::new());
        self.erb_gru.step(&emb, &mut st.h_erb, &mut d);
        self.df_gru.step(&emb, &mut st.h_df, &mut c);
        push(&mut st.c0_hist, &enc.c0);
        self.decode(&enc, &hist(&st.c0_hist), &emb, &d, &c)
    }

    /// Encodeur sans état récurrent : `erb` et `spec` sont les fenêtres causales des convolutions.
    fn encode(&self, erb: &[&[f32]], spec: &[&[f32]]) -> Encoded {
        let (mut e0, mut e1, mut e2, mut e3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        self.erb_conv[0].forward(erb, NB_ERB, &mut e0);
        self.erb_conv[1].forward(&[&e0], NB_ERB, &mut e1);
        self.erb_conv[2].forward(&[&e1], NB_ERB / 2, &mut e2);
        self.erb_conv[3].forward(&[&e2], NB_ERB / 4, &mut e3);
        let (mut c0, mut c1) = (Vec::new(), Vec::new());
        self.df_conv0.forward(spec, NB_DF, &mut c0);
        self.df_conv1.forward(&[&c0], NB_DF, &mut c1);

        // [C][F] → [F][C] aplati, comme `permute(0, 2, 3, 1).flatten(2)`.
        let cemb_in = to_fc(&c1, CH, NB_DF / 2);
        let mut cemb = vec![0.0; EMB];
        self.df_fc_emb.forward(&cemb_in, &mut cemb);
        let mut emb_in = to_fc(&e3, CH, NB_ERB / 4);
        for (e, c) in emb_in.iter_mut().zip(&cemb) {
            *e += c.max(0.0);
        }
        // Branches latérales du U-Net ERB : ne dépendent que de l'encodeur.
        let f8 = NB_ERB / 4;
        let mut skips: [Vec<f32>; 4] = Default::default();
        self.convp[0].forward(&[&e0], NB_ERB, &mut skips[0]);
        self.convp[1].forward(&[&e1], NB_ERB / 2, &mut skips[1]);
        self.convp[2].forward(&[&e2], f8, &mut skips[2]);
        self.convp[3].forward(&[&e3], f8, &mut skips[3]);
        Encoded {
            gates: self.enc_gru.input_gates(&emb_in),
            skips,
            c0,
        }
    }

    /// Décodeurs après les GRU : `c0` est la fenêtre causale de `df_convp`.
    fn decode(&self, enc: &Encoded, c0: &[&[f32]], emb: &[f32], d: &[f32], c: &[f32]) -> StepOut {
        let lsnr = sigmoid(self.lsnr_fc.forward(emb)) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN;

        // Décodeur ERB (U-Net) → masque [NB_ERB].
        let f8 = NB_ERB / 4;
        let mut x = from_fc(d, CH, f8);
        let mut tmp = Vec::new();
        for (level, f) in [f8, f8, NB_ERB / 2].into_iter().enumerate() {
            tmp.clone_from(&enc.skips[3 - level]);
            add(&mut tmp, &x);
            self.convt[2 - level].forward(&[&tmp], f, &mut x);
        }
        tmp.clone_from(&enc.skips[0]);
        add(&mut tmp, &x);
        let mut mask = Vec::new();
        self.conv0_out.forward(&[&tmp], NB_ERB, &mut mask);

        // Décodeur DF → coefficients complexes [NB_DF][ordre][2].
        let mut c = c.to_vec();
        let mut skip = vec![0.0; HIDDEN];
        self.df_skip.forward(emb, &mut skip);
        add(&mut c, &skip);
        let mut coefs = vec![0.0; NB_DF * DF_OUT_CH];
        self.df_out.forward(&c, &mut coefs);
        let mut pathway = Vec::new();
        self.df_convp.forward(c0, NB_DF, &mut pathway);
        for f in 0..NB_DF {
            for o in 0..DF_OUT_CH {
                let v = &mut coefs[f * DF_OUT_CH + o];
                *v = v.tanh() + pathway[o * NB_DF + f];
            }
        }
        StepOut { mask, coefs, lsnr }
    }
}

/// Sorties de l'encodeur d'une trame, avant le GRU.
struct Encoded {
    /// Portes d'entrée de la première couche du GRU d'encodeur.
    gates: Vec<f32>,
    /// `convp[i]` appliquée à la sortie `i` de l'encodeur ERB.
    skips: [Vec<f32>; 4],
    /// Sortie de `df_conv0`, entrée de `df_convp`.
    c0: Vec<f32>,
}

fn hist(h: &[Vec<f32>]) -> Vec<&[f32]> {
    h.iter().map(Vec::as_slice).collect()
}

fn push(hist: &mut [Vec<f32>], frame: &[f32]) {
    hist.rotate_left(1);
    if let Some(last) = hist.last_mut() {
        last.copy_from_slice(frame);
    }
}

fn add(dst: &mut [f32], src: &[f32]) {
    dst.iter_mut().zip(src).for_each(|(d, s)| *d += s);
}

fn to_fc(x: &[f32], c: usize, f: usize) -> Vec<f32> {
    let mut y = vec![0.0; c * f];
    for ci in 0..c {
        for fi in 0..f {
            y[fi * c + ci] = x[ci * f + fi];
        }
    }
    y
}

fn from_fc(x: &[f32], c: usize, f: usize) -> Vec<f32> {
    let mut y = vec![0.0; c * f];
    for ci in 0..c {
        for fi in 0..f {
            y[ci * f + fi] = x[fi * c + ci];
        }
    }
    y
}

/// `DF_ORDER` exposé au filtrage.
pub(crate) const ORDER: usize = DF_ORDER;
