//! Graphe DeepFilterNet3 (`df/deepfilternet3.py`) : encodeur, décodeur ERB et décodeur DF.

use std::collections::HashMap;

use super::dsp::{NB_DF, NB_ERB};
use super::layers::{take_shape, Act, Conv, ConvSpec, GroupedLinear, Gru, Linear1, Math};
use crate::{InferError, Result, Tensor};

mod batch;

pub(super) const CH: usize = 64;
pub(super) const EMB: usize = CH * NB_ERB / 4; // 512
/// Largeur des GRU de DFN3 standard, seule portée sur GPU.
pub(super) const HIDDEN: usize = 256;
const DF_ORDER: usize = 5;
pub(super) const DF_OUT_CH: usize = 2 * DF_ORDER;
const LSNR_MIN: f32 = -15.0;
const LSNR_MAX: f32 = 35.0;

/// Hyperparamètres d'une variante DFN3 (`config.ini`) qui changent le graphe.
///
/// Lus dans les métadonnées safetensors écrites par
/// `tools/voice/convert_dfn3_onnx.py` ; un fichier sans métadonnées (checkpoint
/// converti par `convert_dfn3.py`) est le DFN3 standard.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Config {
    /// `emb_hidden_dim` = `df_hidden_dim` : largeur des trois GRU.
    pub hidden: usize,
    /// `emb_num_layers − 1` : couches du GRU du décodeur ERB.
    pub erb_layers: usize,
    /// `df_num_layers` : couches du GRU du décodeur DF.
    pub df_layers: usize,
    /// `conv_kernel[0]` : noyau temporel des convolutions internes.
    pub conv_kt: usize,
    /// `enc_linear_groups` : groupes de `df_fc_emb`.
    pub enc_groups: usize,
    /// `conv_lookahead` = `df_lookahead`, en trames.
    pub lookahead: usize,
}

impl Config {
    /// DeepFilterNet3 publié (`DeepFilterNet3/config.ini`).
    pub(crate) const STD: Self = Self {
        hidden: HIDDEN,
        erb_layers: 2,
        df_layers: 2,
        conv_kt: 1,
        enc_groups: 32,
        lookahead: 2,
    };

    /// Lit la variante dans les métadonnées ; refuse ce que le port ne couvre pas.
    pub(crate) fn from_metadata(meta: &HashMap<String, String>) -> Result<Self> {
        if !meta.contains_key("conv_lookahead") {
            return Ok(Self::STD);
        }
        let list = |key: &str| -> Result<Vec<usize>> {
            let raw = meta
                .get(key)
                .ok_or_else(|| InferError::Config(format!("métadonnée DFN3 absente : {key}")))?;
            raw.split(',')
                .map(|v| {
                    v.trim().parse().map_err(|_| {
                        InferError::Config(format!("métadonnée DFN3 {key} invalide : {raw}"))
                    })
                })
                .collect()
        };
        let one = |key: &str| -> Result<usize> {
            match list(key)?.as_slice() {
                [v] => Ok(*v),
                _ => Err(InferError::Config(format!(
                    "métadonnée DFN3 {key} : une valeur attendue"
                ))),
            }
        };
        // Constantes du port (STFT, bandes, convolutions d'entrée) : refusées si elles changent.
        let fixed: [(&str, &[usize]); 11] = [
            ("sr", &[48_000]),
            ("fft_size", &[super::dsp::FFT_SIZE]),
            ("hop_size", &[super::dsp::HOP_SIZE]),
            ("nb_erb", &[NB_ERB]),
            ("nb_df", &[NB_DF]),
            ("df_order", &[DF_ORDER]),
            ("conv_ch", &[CH]),
            ("conv_kernel_inp", &[3, 3]),
            ("linear_groups", &[16]),
            ("df_pathway_kernel_size_t", &[5]),
            ("emb_num_layers", &[3]),
        ];
        for (key, want) in fixed {
            if list(key)? != want {
                return Err(InferError::Config(format!(
                    "variante DFN3 non prise en charge : {key} = {:?}",
                    meta.get(key)
                )));
            }
        }
        let (hidden, df_hidden) = (one("emb_hidden_dim")?, one("df_hidden_dim")?);
        let (lookahead, df_lookahead) = (one("conv_lookahead")?, one("df_lookahead")?);
        let kernel = list("conv_kernel")?;
        let (Some(&conv_kt), Some(&3)) = (kernel.first(), kernel.get(1)) else {
            return Err(InferError::Config(format!(
                "conv_kernel DFN3 inattendu : {kernel:?}"
            )));
        };
        if hidden != df_hidden || lookahead != df_lookahead || lookahead >= DF_ORDER || conv_kt == 0
        {
            return Err(InferError::Config(format!(
                "variante DFN3 non prise en charge : hidden {hidden}/{df_hidden}, \
                 lookahead {lookahead}/{df_lookahead}, conv_kernel {kernel:?}"
            )));
        }
        Ok(Self {
            hidden,
            erb_layers: 2,
            df_layers: one("df_num_layers")?,
            conv_kt,
            enc_groups: one("enc_linear_groups")?,
            lookahead,
        })
    }
}

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
pub(super) struct SqueezedGru {
    pub(super) lin_in: GroupedLinear,
    pub(super) layers: Vec<Gru>,
    pub(super) lin_out: Option<GroupedLinear>,
}

impl SqueezedGru {
    fn load(
        w: &mut HashMap<String, Tensor>,
        prefix: &str,
        (num_layers, hidden): (usize, usize),
        groups: usize,
        output: Option<usize>,
    ) -> Result<Self> {
        let lin_in = GroupedLinear::load(
            w,
            &format!("{prefix}.linear_in.0.weight"),
            EMB,
            hidden,
            groups,
        )?;
        let layers = (0..num_layers)
            .map(|l| Gru::load(w, &format!("{prefix}.gru"), l, hidden, hidden))
            .collect::<Result<Vec<_>>>()?;
        let lin_out = output
            .map(|o| {
                GroupedLinear::load(
                    w,
                    &format!("{prefix}.linear_out.0.weight"),
                    hidden,
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

    fn step(&self, x: &[f32], h: &mut [Vec<f32>], out: &mut Vec<f32>, math: Math) {
        self.recur(&self.input_gates(x), h, out, math);
    }

    fn hidden(&self) -> usize {
        self.layers[0].hidden
    }

    /// `linear_in`, ReLU puis portes d'entrée de la première couche : sans état.
    fn input_gates(&self, x: &[f32]) -> Vec<f32> {
        let mut a = vec![0.0; self.hidden()];
        self.lin_in.forward(x, &mut a);
        a.iter_mut().for_each(|v| *v = v.max(0.0));
        self.layers[0].input_gates(&a)
    }

    /// Reste du pas, récurrent, à partir des portes de [`Self::input_gates`].
    fn recur(&self, gi0: &[f32], h: &mut [Vec<f32>], out: &mut Vec<f32>, math: Math) {
        let mut a = vec![0.0; self.hidden()];
        for (l, (layer, state)) in self.layers.iter().zip(h.iter_mut()).enumerate() {
            if l == 0 {
                layer.recur(gi0, state, math);
            } else {
                layer.step(&a, state, math);
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
    pub(super) erb_conv: [Conv; 4],
    pub(super) df_conv0: Conv,
    pub(super) df_conv1: Conv,
    pub(super) df_fc_emb: GroupedLinear,
    pub(super) enc_gru: SqueezedGru,
    pub(super) lsnr_fc: Linear1,
    pub(super) erb_gru: SqueezedGru,
    pub(super) convp: [Conv; 4],
    pub(super) convt: [Conv; 3],
    pub(super) conv0_out: Conv,
    pub(super) df_convp: Conv,
    pub(super) df_gru: SqueezedGru,
    pub(super) df_skip: GroupedLinear,
    pub(super) df_out: GroupedLinear,
    /// `[NB_ERB][481]`, matrice inverse 0/1 du masque ERB.
    pub(crate) erb_inv: Vec<f32>,
    /// `[481][NB_ERB]`, banque ERB normalisée (sert à vérifier les largeurs).
    pub(crate) erb_fb: Vec<f32>,
    pub(crate) config: Config,
}

/// État récurrent et historiques temporels d'un flux.
pub(crate) struct NetState {
    erb_hist: Vec<Vec<f32>>,
    spec_hist: Vec<Vec<f32>>,
    c0_hist: Vec<Vec<f32>>,
    h_enc: Vec<Vec<f32>>,
    h_erb: Vec<Vec<f32>>,
    h_df: Vec<Vec<f32>>,
    /// Entrées passées des convolutions internes à noyau temporel > 1 (variante
    /// faible latence) ; vides en DFN3 standard.
    conv_hist: ConvHist,
    pub(crate) math: Math,
}

/// Fenêtres causales des convolutions internes, indexées par les constantes `H_*`.
#[derive(Default)]
pub(crate) struct ConvHist([Vec<Vec<f32>>; 6]);

/// Emplacements de [`ConvHist`] : entrées de `erb_conv1..3`, `df_conv1`, `convt3`, `conv0_out`.
const H_ERB1: usize = 0;
const H_DF1: usize = 3;
const H_CONVT3: usize = 4;
const H_OUT: usize = 5;

impl ConvHist {
    /// Applique `conv` à la trame `x` précédée de ses `kt − 1` entrées passées.
    fn forward(
        &mut self,
        slot: usize,
        conv: &Conv,
        x: &[f32],
        fin: usize,
        out: &mut Vec<f32>,
        math: Math,
    ) {
        if conv.kt() == 1 {
            conv.forward(&[x], fin, out, math);
            return;
        }
        let h = &mut self.0[slot];
        if h.len() != conv.kt() {
            *h = vec![vec![0.0; x.len()]; conv.kt()];
        }
        push(h, x);
        conv.forward(&hist(h), fin, out, math);
    }
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
    pub(crate) fn load(
        mut w: HashMap<String, Tensor>,
        freq_bins: usize,
        config: Config,
    ) -> Result<Self> {
        use Act::{Relu, Sigmoid};
        let w = &mut w;
        let (kt, hidden) = (config.conv_kt, config.hidden);
        let erb_conv = [
            Conv::load(
                w,
                &spec("enc.erb_conv0", 1, CH, 1, 3, 3, 1, false, false, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv1", CH, CH, CH, kt, 3, 2, false, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv2", CH, CH, CH, kt, 3, 2, false, true, Relu),
            )?,
            Conv::load(
                w,
                &spec("enc.erb_conv3", CH, CH, CH, kt, 3, 1, false, true, Relu),
            )?,
        ];
        let df_conv0 = Conv::load(
            w,
            &spec("enc.df_conv0", 2, CH, 2, 3, 3, 1, false, true, Relu),
        )?;
        let df_conv1 = Conv::load(
            w,
            &spec("enc.df_conv1", CH, CH, CH, kt, 3, 2, false, true, Relu),
        )?;
        let df_fc_emb = GroupedLinear::load(
            w,
            "enc.df_fc_emb.0.weight",
            CH * NB_DF / 2,
            EMB,
            config.enc_groups,
        )?;
        let enc_gru = SqueezedGru::load(w, "enc.emb_gru", (1, hidden), 16, Some(EMB))?;
        let lsnr_fc = Linear1::load(w, "enc.lsnr_fc.0", EMB)?;
        let erb_gru = SqueezedGru::load(
            w,
            "erb_dec.emb_gru",
            (config.erb_layers, hidden),
            16,
            Some(EMB),
        )?;
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
                &spec("erb_dec.convt3", CH, CH, CH, kt, 3, 1, false, true, Relu),
            )?,
        ];
        let conv0_out = Conv::load(
            w,
            &spec(
                "erb_dec.conv0_out",
                CH,
                1,
                1,
                kt,
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
        let df_gru = SqueezedGru::load(w, "df_dec.df_gru", (config.df_layers, hidden), 8, None)?;
        let df_skip = GroupedLinear::load(w, "df_dec.df_skip.weight", EMB, hidden, 16)?;
        let df_out =
            GroupedLinear::load(w, "df_dec.df_out.0.weight", hidden, NB_DF * DF_OUT_CH, 16)?;
        // L'export ONNX n'embarque pas la banque ERB : elle se déduit des largeurs libDF.
        let (erb_inv, erb_fb) = if w.contains_key("erb_fb") {
            (
                take_shape(w, "mask.erb_inv_fb", &[NB_ERB, freq_bins])?,
                take_shape(w, "erb_fb", &[freq_bins, NB_ERB])?,
            )
        } else {
            erb_banks(&super::dsp::erb_widths(), freq_bins)
        };
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
            config,
        })
    }

    /// Variante standard : seule celle-ci a un chemin GPU et un pas multi-cœur.
    pub(crate) fn is_std(&self) -> bool {
        self.config == Config::STD
    }

    pub(crate) fn new_state(&self) -> NetState {
        let zeros = |n: usize, len: usize| vec![vec![0.0_f32; len]; n];
        NetState {
            erb_hist: zeros(self.erb_conv[0].kt(), NB_ERB),
            spec_hist: zeros(self.df_conv0.kt(), 2 * NB_DF),
            c0_hist: zeros(self.df_convp.kt(), CH * NB_DF),
            h_enc: zeros(1, self.config.hidden),
            h_erb: zeros(self.config.erb_layers, self.config.hidden),
            h_df: zeros(self.config.df_layers, self.config.hidden),
            conv_hist: ConvHist::default(),
            math: Math::Libm,
        }
    }

    /// Un pas du réseau sur les features décalées `s[τ] = feat[τ + conv_lookahead]`.
    pub(crate) fn step(&self, st: &mut NetState, erb: &[f32], cplx: &[f32]) -> StepOut {
        let math = st.math;
        push(&mut st.erb_hist, erb);
        push(&mut st.spec_hist, cplx);
        let enc = self.encode(
            &hist(&st.erb_hist),
            &hist(&st.spec_hist),
            &mut st.conv_hist,
            math,
        );
        let mut emb = Vec::new();
        self.enc_gru
            .recur(&enc.gates, &mut st.h_enc, &mut emb, math);
        let (mut d, mut c) = (Vec::new(), Vec::new());
        self.erb_gru.step(&emb, &mut st.h_erb, &mut d, math);
        self.df_gru.step(&emb, &mut st.h_df, &mut c, math);
        push(&mut st.c0_hist, &enc.c0);
        self.decode(
            &enc,
            &hist(&st.c0_hist),
            &emb,
            (&d, &c),
            &mut st.conv_hist,
            math,
        )
    }

    /// Encodeur sans état récurrent : `erb` et `spec` sont les fenêtres causales des convolutions.
    /// `ch` ne sert qu'aux convolutions internes à noyau temporel > 1.
    fn encode(&self, erb: &[&[f32]], spec: &[&[f32]], ch: &mut ConvHist, math: Math) -> Encoded {
        let (mut e0, mut e1, mut e2, mut e3) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        self.erb_conv[0].forward(erb, NB_ERB, &mut e0, math);
        ch.forward(H_ERB1, &self.erb_conv[1], &e0, NB_ERB, &mut e1, math);
        ch.forward(
            H_ERB1 + 1,
            &self.erb_conv[2],
            &e1,
            NB_ERB / 2,
            &mut e2,
            math,
        );
        ch.forward(
            H_ERB1 + 2,
            &self.erb_conv[3],
            &e2,
            NB_ERB / 4,
            &mut e3,
            math,
        );
        let (mut c0, mut c1) = (Vec::new(), Vec::new());
        self.df_conv0.forward(spec, NB_DF, &mut c0, math);
        ch.forward(H_DF1, &self.df_conv1, &c0, NB_DF, &mut c1, math);

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
        self.convp[0].forward(&[&e0], NB_ERB, &mut skips[0], math);
        self.convp[1].forward(&[&e1], NB_ERB / 2, &mut skips[1], math);
        self.convp[2].forward(&[&e2], f8, &mut skips[2], math);
        self.convp[3].forward(&[&e3], f8, &mut skips[3], math);
        Encoded {
            gates: self.enc_gru.input_gates(&emb_in),
            skips,
            c0,
        }
    }

    /// Décodeurs après les GRU : `c0` est la fenêtre causale de `df_convp`.
    fn decode(
        &self,
        enc: &Encoded,
        c0: &[&[f32]],
        emb: &[f32],
        (d, c): (&[f32], &[f32]),
        ch: &mut ConvHist,
        math: Math,
    ) -> StepOut {
        let lsnr = math.sigmoid(self.lsnr_fc.forward(emb)) * (LSNR_MAX - LSNR_MIN) + LSNR_MIN;

        // Décodeur ERB (U-Net) → masque [NB_ERB].
        let f8 = NB_ERB / 4;
        let mut x = from_fc(d, CH, f8);
        let mut tmp = Vec::new();
        for (level, f) in [f8, f8, NB_ERB / 2].into_iter().enumerate() {
            tmp.clone_from(&enc.skips[3 - level]);
            add(&mut tmp, &x);
            if level == 0 {
                ch.forward(H_CONVT3, &self.convt[2], &tmp, f, &mut x, math);
            } else {
                self.convt[2 - level].forward(&[&tmp], f, &mut x, math);
            }
        }
        tmp.clone_from(&enc.skips[0]);
        add(&mut tmp, &x);
        let mut mask = Vec::new();
        ch.forward(H_OUT, &self.conv0_out, &tmp, NB_ERB, &mut mask, math);

        // Décodeur DF → coefficients complexes [NB_DF][ordre][2].
        let mut c = c.to_vec();
        let mut skip = vec![0.0; c.len()];
        self.df_skip.forward(emb, &mut skip);
        add(&mut c, &skip);
        let mut coefs = vec![0.0; NB_DF * DF_OUT_CH];
        self.df_out.forward(&c, &mut coefs);
        let mut pathway = Vec::new();
        self.df_convp.forward(c0, NB_DF, &mut pathway, math);
        for f in 0..NB_DF {
            for o in 0..DF_OUT_CH {
                let v = &mut coefs[f * DF_OUT_CH + o];
                *v = math.tanh(*v) + pathway[o * NB_DF + f];
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

/// Banques ERB `[NB_ERB][bins]` (inverse 0/1) et `[bins][NB_ERB]` (moyenne), comme libDF.
fn erb_banks(widths: &[usize], bins: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut inv, mut fb) = (vec![0.0; NB_ERB * bins], vec![0.0; bins * NB_ERB]);
    let mut start = 0;
    for (band, &width) in widths.iter().enumerate() {
        for f in start..(start + width).min(bins) {
            inv[band * bins + f] = 1.0;
            fb[f * NB_ERB + band] = 1.0 / width as f32;
        }
        start += width;
    }
    (inv, fb)
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
